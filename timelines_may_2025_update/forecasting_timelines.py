import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde
import matplotlib.pyplot as plt
import yaml
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import matplotlib.font_manager as fm

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_lognormal_from_80_ci(lower_bound, upper_bound):
    # Given 80% CI (10th and 90th percentiles)
    # Convert to natural log space
    ln_lower = np.log(lower_bound)
    ln_upper = np.log(upper_bound)

    # Z-scores for 10th and 90th percentiles
    z_low = norm.ppf(0.1)  # ≈ -1.28
    z_high = norm.ppf(0.9)  # ≈ 1.28

    # Calculate mu and sigma in log space
    sigma = (ln_upper - ln_lower) / (z_high - z_low)
    mu = (ln_upper + ln_lower) / 2

    # Return the lognormal distribution
    return lognorm(s=sigma, scale=np.exp(mu))


def get_normal_from_80_ci(lower_bound, upper_bound):
    # Z-scores for 10th and 90th percentiles
    z_low = norm.ppf(0.1)  # ≈ -1.28
    z_high = norm.ppf(0.9)  # ≈ 1.28

    # Calculate mu and sigma
    mu = (upper_bound + lower_bound) / 2
    sigma = (upper_bound - lower_bound) / (z_high - z_low)

    return norm(loc=mu, scale=sigma)


def weighted_geometric_mean(values, weights):
    r"""Calculate weighted geometric mean using the formula:
    \bar{x}^G = \exp(\frac{1}{\sum w_i} \sum w_i \ln x_i)
    """
    weights_sum = sum(weights)
    normalized_weights = [w / weights_sum for w in weights]
    return np.exp(sum(w * np.log(x) for w, x in zip(normalized_weights, values)))

def get_distribution_samples(config: dict, n_sims: int, correlation: float = 0.7, t_start: float = None) -> dict:
    """Generate correlated samples from all input distributions."""
    # First generate correlated standard normal variables
    n_vars = 3 + len(config["algorithmic_slowdowns"])  # Core params (excluding h_sat, h_SC, d, v_algorithmic vars) + A_values
    
    # Create correlation matrix (all pairs have same correlation)
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated standard normal samples
    mean = np.zeros(n_vars)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_sims)
    
    # Convert to uniform using the probability integral transform
    uniform_samples = norm.cdf(normal_samples)
    
    # Now convert to desired distributions using inverse CDF
    samples = {}
    idx = 0
    
    # Sample initial software progress share from normal distribution
    lower, upper = config["initial_software_progress_share_ci"]
    # Convert 80% CI to normal distribution parameters
    z_low = -1.28  # norm.ppf(0.1)
    z_high = 1.28  # norm.ppf(0.9)
    mean = (lower + upper) / 2
    std = (upper - lower) / (z_high - z_low)
    # Generate samples and clip to [0.1, 0.9]
    samples["initial_software_progress_share"] = np.clip(
        np.random.normal(mean, std, n_sims),
        0.1, 0.9
    )
    
    # Sample h_sat independently
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_sat_ci"][0],  # In hours
        config["distributions"]["h_sat_ci"][1]
    )
    samples["h_sat"] = dist.ppf(np.random.random(n_sims)) / (24 * 30)  # Convert hours to months
    
    # Sample h_SC independently - already in work months
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_SC_ci"][0],  # In work months
        config["distributions"]["h_SC_ci"][1]
    )
    samples["h_SC"] = dist.ppf(np.random.random(n_sims))  # Already in months

    # Generate separate correlated samples for progress multipliers
    n_prog_vars = 3  # v_present_day, v_software_sat, v_software_SC
    prog_corr_matrix = np.array([[1.0, correlation, correlation], 
                                [correlation, 1.0, correlation], 
                                [correlation, correlation, 1.0]])
    prog_normal_samples = np.random.multivariate_normal(np.zeros(n_prog_vars), prog_corr_matrix, size=n_sims)
    prog_uniform_samples = norm.cdf(prog_normal_samples)
    
    # Sample present day progress multiplier first
    dist = get_lognormal_from_80_ci(*config["distributions"]["present_day_prog_multiplier_ci"])
    samples["v_present_day"] = dist.ppf(prog_uniform_samples[:, 0])
    
    # Sample v_software_sat, ensuring it's always greater than present day
    dist_sat = get_lognormal_from_80_ci(*config["distributions"]["v_algorithmic_sat_ci"])
    v_sat_raw = dist_sat.ppf(prog_uniform_samples[:, 1])
    # Ensure v_software_sat > v_present_day
    samples["v_software_sat"] = np.maximum(samples["v_present_day"], v_sat_raw)
    
    dist = get_lognormal_from_80_ci(*config["distributions"]["v_algorithmic_SC_ci"])
    v_sc_raw = dist.ppf(prog_uniform_samples[:, 2])
    # Ensure v_software_SC > v_software_sat
    samples["v_software_SC"] = np.maximum(samples["v_software_sat"], v_sc_raw)

    # Sample horizon_doubling_time with correlation
    dist = get_lognormal_from_80_ci(*config["distributions"]["horizon_doubling_time_ci"])
    samples["horizon_doubling_time"] = dist.ppf(uniform_samples[:, idx])
    idx += 1

    # Handle t_sat as dates - use t_start instead of current date
    if t_start is not None:
        # Convert t_start to year and month for date calculations
        start_year = int(t_start)
        start_month = int((t_start % 1) * 12) + 1
        reference_date = datetime(start_year, start_month, 1)
    else:
        reference_date = datetime.now()
    
    date1 = datetime.strptime(config["distributions"]["t_sat_ci"][0], "%Y-%m-%d")
    date2 = datetime.strptime(config["distributions"]["t_sat_ci"][1], "%Y-%m-%d")
    months1 = (date1.year - reference_date.year) * 12 + date1.month - reference_date.month
    months2 = (date2.year - reference_date.year) * 12 + date2.month - reference_date.month
    
    # Create lognormal distribution for months until saturation
    dist = get_lognormal_from_80_ci(months1, months2)
    samples["t_sat"] = dist.ppf(uniform_samples[:, idx])  # In months
    idx += 1
    
    # Sample d independently
    dist = get_lognormal_from_80_ci(*config["distributions"]["d_ci"])
    samples["d"] = dist.ppf(np.random.random(n_sims))  # Already in months
    
    # Store the superexponential schedule for later use
    samples["superexponential_schedule_months"] = config["distributions"]["superexponential_schedule_months"]
    
    # Sample subexponential probability
    p_sub = config["distributions"]["p_subexponential"]
    
    # Generate independent uniform samples for growth type
    growth_type = np.random.uniform(0, 1, n_sims)
    samples["is_subexponential"] = growth_type > (1 - p_sub)
    samples["is_exponential"] = ~samples["is_subexponential"]
    
    # For each simulation, determine if and when it becomes superexponential
    samples["superexponential_start_time"] = np.full(n_sims, np.inf)  # Default to never becoming superexponential
    for i in range(n_sims):
        if not samples["is_subexponential"][i]:  # Only consider non-subexponential cases
            # Generate a random number to determine if/when it becomes superexponential
            for horizon, prob in samples["superexponential_schedule_months"]:
                if growth_type[i] < prob:
                    # Store the horizon directly since it's already in months
                    samples["superexponential_start_time"][i] = horizon
                    break
    
    # Sample se_doubling_decay_fraction from lognormal distribution
    dist = get_lognormal_from_80_ci(
        config["distributions"]["se_doubling_decay_fraction_ci"][0],
        config["distributions"]["se_doubling_decay_fraction_ci"][1]
    )
    samples["se_doubling_decay_fraction"] = np.clip(dist.rvs(n_sims), 0, 1)  # Clip to ensure decay is between 0 and 1
    
    # Add subexponential growth parameter
    samples["sub_doubling_growth_fraction"] = config["distributions"]["sub_doubling_growth_fraction"]
    
    # Algorithmic slowdowns with probability of being zero
    samples["A_values"] = {}
    for name, params in config["algorithmic_slowdowns"].items():
        p_zero, lower, upper = params
        
        # Generate uniform random numbers for determining if slowdown is zero
        is_nonzero = np.random.random(n_sims) >= p_zero
        
        # Generate lognormal samples for non-zero cases
        dist = get_lognormal_from_80_ci(lower, upper)
        values = dist.ppf(uniform_samples[:, idx])
        
        # Set values to zero according to p_zero
        values[~is_nonzero] = 0
        
        samples["A_values"][name] = values  # Already in months
        idx += 1
    
    return samples

def get_v_compute(t: float, compute_schedule: list) -> float:
    """Calculate compute progress rate based on time and compute schedule.
    
    Args:
        t: Current time in years
        compute_schedule: List of [year, rate] pairs, sorted by year
    """
    # Default rate is 1.0
    current_rate = 1.0
    
    # Find the most recent schedule entry that applies
    for year, rate in compute_schedule:
        if t >= year:
            current_rate = rate
        else:
            break
            
    return current_rate

def calculate_gaps(samples: dict) -> tuple[np.ndarray, np.ndarray]:
    """Calculate horizon gap and total algorithmic gap."""
    # Calculate number of doublings needed
    n_doublings = np.log2(samples["h_SC"]/samples["h_sat"])
    
    # Calculate horizon gap based on growth type
    g_h = np.zeros_like(n_doublings)
    
    # For each simulation, determine if it becomes superexponential during the gap
    for i in range(len(n_doublings)):
        if samples["is_subexponential"][i]:
            # For subexponential cases
            # Each doubling takes (1+growth)^k times as long as the first doubling
            growth = samples["sub_doubling_growth_fraction"]
            first_doubling_time = samples["horizon_doubling_time"][i]
            n = n_doublings[i]
            ratio = 1 + growth
            
            # Use geometric series sum formula: T * (r^n-1)/(r-1)
            # where T is first doubling time, r is (1+growth), n is number of doublings
            g_h[i] = first_doubling_time * (ratio**n - 1) / (ratio - 1)
        else:
            # For non-subexponential cases, check if/when it becomes superexponential
            superexponential_start = samples["superexponential_start_time"][i]
            if superexponential_start < np.inf:
                # Calculate how many doublings happen before superexponential transition
                doublings_before = np.log2(superexponential_start/samples["h_sat"][i])
                doublings_before = min(doublings_before, n_doublings[i])
                doublings_after = n_doublings[i] - doublings_before
                
                # Calculate time for exponential phase
                first_doubling_time = samples["horizon_doubling_time"][i]
                g_h[i] = first_doubling_time * doublings_before
                
                # Calculate time for superexponential phase
                if doublings_after > 0:
                    decay = samples["se_doubling_decay_fraction"][i]  # Get decay for this specific simulation
                    ratio = 1 - decay
                    # Use geometric series sum formula for remaining doublings
                    g_h[i] += first_doubling_time * (1 - ratio**doublings_after) / (1 - ratio)
            else:
                # Pure exponential case
                g_h[i] = samples["horizon_doubling_time"][i] * n_doublings[i]
    
    # Calculate total algorithmic gap including all slowdowns
    A_values = list(samples["A_values"].values())
    all_gaps = [g_h] + A_values
    sum_gaps = np.sum(all_gaps, axis=0)
    g_SC = sum_gaps
    
    return g_h, g_SC

def get_labor_growth_rate(t: float, labor_growth_schedule: list) -> float:
    """Calculate labor growth rate based on time and labor growth schedule.
    
    Args:
        t: Current time in years
        labor_growth_schedule: List of [year, rate] pairs, sorted by year
    """
    # Default rate is 0.5 (same as before)
    current_rate = 0.5
    
    # Find the most recent schedule entry that applies
    for year, rate in labor_growth_schedule:
        if t >= year:
            current_rate = rate
        else:
            break
            
    return current_rate

def run_single_forecaster_simulation(samples: dict, params: dict, forecaster_config: dict, simulation_config: dict) -> tuple[list[float], list[list[float]], list[list[float]]]:
    """Run simulation for a single forecaster configuration."""
    successful_times = []
    research_trajectories = []  # Store research stock over time for each simulation
    time_trajectories = []      # Store corresponding time points
    _, g_SC = calculate_gaps(samples)  # Use sampled h_sat values
    
    # Get software progress share from samples
    software_progress_share = samples["initial_software_progress_share"]
    
    for i in tqdm(range(len(samples["horizon_doubling_time"])), desc="Running simulations", leave=False):
        # First, simulate the pre-saturation period
        stock_at_sat, labor_pool_at_sat, pre_sat_research_traj, pre_sat_time_traj = simulate_pre_saturation_period(
            samples, i, params, forecaster_config, simulation_config
        )
        
        # Initialize post-saturation simulation
        t = params["t_0"] + samples["t_sat"][i]/12 - samples["d"][i]/12  # Convert months to years
        g_t = 0
        
        # Use the labor pool from pre-saturation simulation
        current_labor_pool = labor_pool_at_sat
        current_research_stock = stock_at_sat
        
        # Start with the pre-saturation trajectory
        sim_research_trajectory = pre_sat_research_traj.copy()
        sim_time_trajectory = pre_sat_time_traj.copy()
        
        # Run post-saturation timesteps
        max_time = simulation_config["max_time"]  # Get max_time from simulation config
        
        for _ in range(params["n_steps"]):
            # Calculate progress fraction
            progress_fraction = g_t / g_SC[i]
            
            # Calculate software progress rate based on intermediate speedup (interpolate between sat and SC rates)
            software_prog_multiplier = (1 + samples["v_software_sat"][i]) * ((1 + samples["v_software_SC"][i])/(1 + samples["v_software_sat"][i])) ** progress_fraction

            # Get current labor growth rate from schedule
            current_labor_growth_rate = get_labor_growth_rate(t, forecaster_config["labor_growth_schedule"])
            
            # Convert annual growth rate to daily rate for the time step
            daily_growth_rate = (1 + current_labor_growth_rate) ** (params["dt"]/250) - 1

            # Calculate new labor added this period
            new_labor = current_labor_pool * daily_growth_rate
            current_labor_pool += new_labor
            
            # Calculate research contribution on a yearly basis, then divide
            research_contribution = ((((current_labor_pool+1) ** simulation_config["labor_power"])-1) * software_prog_multiplier) / (250/params["dt"])

            # Add to research stock
            new_research_stock = current_research_stock + research_contribution
            
            # Calculate actual growth rate (annualized)
            actual_growth = (new_research_stock / current_research_stock) ** (250/params["dt"]) - 1

            if g_t == 0:
                baseline_growth = actual_growth
            
            # Calculate adjustment factor based on growth rate ratio
            # Using log ratio to properly account for compound growth
            growth_ratio = np.log(1 + actual_growth) / np.log(1 + baseline_growth)

            # Get compute progress rate using compute schedule
            v_compute = get_v_compute(t, forecaster_config["compute_schedule"])
            
            # Calculate total progress rate using weighted average
            v_t = software_progress_share[i] * growth_ratio + (1 - software_progress_share[i]) * v_compute
            
            # Update progress (dt is in days, convert to months by dividing by ~30.5)
            g_t += v_t * (params["dt"]/30.5)
            
            if g_t >= g_SC[i]:
                successful_times.append(t)
                break
                
            t += params["dt"]/365  # Convert days to years

            # Check if we've exceeded maximum time
            if t >= max_time:
                successful_times.append(max_time)
                break
            
            # Update research stock and store trajectory (only if continuing)
            current_research_stock = new_research_stock
            sim_research_trajectory.append(current_research_stock)
            sim_time_trajectory.append(t)
        
        # Store complete trajectories for this simulation
        research_trajectories.append(sim_research_trajectory)
        time_trajectories.append(sim_time_trajectory)
            
    return successful_times, research_trajectories, time_trajectories

def setup_plotting_style(plotting_style: dict):
    """Set up matplotlib style according to config."""
    plt.style.use('default')  # Reset to default style
    
    # Set background color to cream (255, 254, 248)
    background_color = plotting_style.get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    plt.rcParams['figure.facecolor'] = bg_rgb
    plt.rcParams['axes.facecolor'] = bg_rgb
    plt.rcParams['savefig.facecolor'] = bg_rgb
    
    # Use monospace font for all text
    font_family = plotting_style.get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family
    
    # Create font properties objects with sizes
    font_regular = fm.FontProperties(family=font_family, size=plotting_style["font"]["sizes"]["title"])
    font_regular_small = fm.FontProperties(family=font_family, size=plotting_style["font"]["sizes"]["axis_labels"])
    font_regular_legend = fm.FontProperties(family=font_family, size=plotting_style["font"]["sizes"]["legend"])
    font_bold = fm.FontProperties(family=font_family, weight='bold', size=plotting_style["font"]["sizes"]["title"])
    font_medium = fm.FontProperties(family=font_family, weight='medium', size=plotting_style["font"]["sizes"]["title"])
    
    # Return the font properties to be used in plotting functions
    return {
        'regular': font_regular,
        'regular_small': font_regular_small,
        'regular_legend': font_regular_legend,
        'bold': font_bold,
        'medium': font_medium
    }

def create_headline_plot(all_forecaster_results: dict[str, list[float]], bins: np.ndarray, bin_width: float, config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create the headline figure showing results with both types of slowdowns for multiple forecasters."""
    background_color = plotting_style.get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(10, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Get current year for x-axis range
    current_year = datetime.now().year
    x_min = current_year
    x_max = current_year + 11  # Show 11 years into the future
    
    for name, results in all_forecaster_results.items():
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from the forecaster's config
        color = config["forecasters"][base_name]["color"]
        
        # Filter out >max_time points for density plot
        max_time = config["simulation"]["max_time"]
        valid_results = [r for r in results if r <= max_time]
        
        # Use KDE for smooth density estimation
        kde = gaussian_kde(valid_results)
        x_range = np.linspace(min(valid_results), max(valid_results), 200)
        density = kde(x_range)
        
        # Plot line with shaded area
        ax.plot(x_range, density, '-', color=color, label=name,
                linewidth=2, alpha=0.8, zorder=2)
        ax.fill_between(x_range, density, color=color, alpha=0.1)
    
    # Configure plot styling with direct fontsize and font
    title = ax.set_title("Superhuman Coder Arrival, Benchmarks and Gaps",
                 fontsize=plotting_style["font"]["sizes"]["title"],
                 pad=10)
    title.set_fontproperties(fonts['regular'])
    
    xlabel = ax.set_xlabel("Year", 
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"],
                 labelpad=10)
    xlabel.set_fontproperties(fonts['regular_small'])
    
    ylabel = ax.set_ylabel("Probability Density", 
                  fontsize=plotting_style["font"]["sizes"]["axis_labels"],
                  labelpad=10)
    ylabel.set_fontproperties(fonts['regular_small'])
    
    # Set axis properties
    ax.set_xticks(range(x_min, x_max + 1))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None)  # Auto-scale upper limit
    
    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add statistics text for each forecaster
    stats_text = []
    for name, results in all_forecaster_results.items():
        # Calculate percentiles, marking >max_time as appropriate
        max_time = config["simulation"]["max_time"]
        p10 = np.percentile(results, 10)
        p50 = np.percentile(results, 50)
        p90 = np.percentile(results, 90)
        
        stats = (
            f"{name}:\n"
            f"  10th: {format_year_month(p10, max_time) if p10 <= max_time else f'>{int(max_time)}'}\n"
            f"  50th: {format_year_month(p50, max_time) if p50 <= max_time else f'>{int(max_time)}'}\n"
            f"  90th: {format_year_month(p90, max_time) if p90 <= max_time else f'>{int(max_time)}'}\n"
        )
        stats_text.append(stats)
    
    text = ax.text(0.7, 0.98, "\n\n".join(stats_text),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=plotting_style["font"]["sizes"]["legend"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Add legend with specific font
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks with specific font
    ax.tick_params(axis="both", 
                   labelsize=plotting_style["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(fonts['regular_legend'])
        if tick in ax.get_xticklabels():
            tick.set_rotation(45)
        
    return fig

def create_scenario_plots(all_forecaster_scenarios: dict[str, list[list[float]]], scenarios: list[tuple],
                         bins: np.ndarray, bin_width: float, config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create figure showing results for all scenarios from multiple forecasters."""
    background_color = plotting_style.get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(15, 6), dpi=150, facecolor=bg_rgb)
    gs = fig.add_gridspec(1, 2, hspace=0.7, wspace=0.3)
    axes = gs.subplots()

    # Get current year for x-axis range
    current_year = datetime.now().year
    x_min = current_year
    x_max = current_year + 11  # Show 11 years into the future
    
    # Update suptitle with direct fontsize and font
    title = plt.suptitle("Date when SC milestone is reached internally",
                 fontsize=plotting_style["font"]["sizes"]["main_title"],
                 y=1.02)
    title.set_fontproperties(fonts['regular'])
    
    for scenario_idx, (_, title) in enumerate(scenarios):
        ax = axes[scenario_idx]
        ax.set_facecolor(bg_rgb)
        
        for name, all_results in all_forecaster_scenarios.items():
            results = all_results[scenario_idx]
            
            # Get color from the forecaster's config
            color = config["forecasters"][name.lower()]["color"]
            
            # Filter out >max_time points for density plot
            max_time = config["simulation"]["max_time"]
            valid_results = [r for r in results if r <= max_time]
            
            # Use KDE for smooth density estimation
            kde = gaussian_kde(valid_results)
            x_range = np.linspace(min(valid_results), max(valid_results), 200)
            density = kde(x_range)
            
            # Plot line with shaded area
            ax.plot(x_range, density, '-', color=color, label=name if scenario_idx == 0 else "",
                    linewidth=2, alpha=0.8, zorder=2)
            ax.fill_between(x_range, density, color=color, alpha=0.1)
        
        # Configure individual subplot with direct fontsize and font
        title = ax.set_title(f"{title}", 
                    fontsize=plotting_style["font"]["sizes"]["title"],
                    pad=5)
        title.set_fontproperties(fonts['regular'])
        
        # Set axis properties
        ax.set_xticks(range(x_min, x_max + 1))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, None)  # Auto-scale upper limit
        
        # Grid and spines
        ax.grid(True, alpha=0.2, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add statistics text for each forecaster
        stats_text = []
        for name, all_results in all_forecaster_scenarios.items():
            results = all_results[scenario_idx]
            # Calculate percentiles, marking >max_time as appropriate
            max_time = config["simulation"]["max_time"]
            p10 = np.percentile(results, 10)
            p50 = np.percentile(results, 50)
            p90 = np.percentile(results, 90)
            
            stats = (
                f"{name}:\n"
                f"  10th: {format_year_month(p10, max_time) if p10 <= max_time else f'>{int(max_time)}'}\n"
                f"  50th: {format_year_month(p50, max_time) if p50 <= max_time else f'>{int(max_time)}'}\n"
                f"  90th: {format_year_month(p90, max_time) if p90 <= max_time else f'>{int(max_time)}'}"
            )
            stats_text.append(stats)
        
        text = ax.text(0.55, 0.98, "\n\n".join(stats_text),
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                fontsize=plotting_style["font"]["sizes"]["legend"])
        text.set_fontproperties(fonts['regular_legend'])
        
        # Configure ticks
        ax.tick_params(axis="both",
                      labelsize=plotting_style["font"]["sizes"]["ticks"])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    
    # Add legend to first subplot only
    axes[0].legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
            
    return fig

def get_pretty_name(raw_name: str) -> str:
    """Convert raw algorithmic gap names to prettier display names."""
    name_map = {
        "project_coordination": "Project Coordination",
        "complex_engineering": "Complex Engineering",
        "feedback_loops": "Feedback Loops",
        "specialization": "Specialization",
        "cost_and_speed": "Cost and Speed",
        "other": "Other"
    }
    return name_map.get(raw_name, raw_name)

def create_distribution_plots(all_forecaster_samples: dict, config: dict, plotting_style: dict, fonts: dict) -> tuple[plt.Figure, plt.Figure]:
    """Create figures showing input distributions and derived quantities for all forecasters."""
    background_color = plotting_style.get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    # Input distributions figure
    fig_inputs = plt.figure(figsize=(20, 15), dpi=150, facecolor=bg_rgb)
    gs = fig_inputs.add_gridspec(4, 4, hspace=1.2, wspace=0.3)
    axes = gs.subplots()
    for ax in axes.flat:
        ax.set_facecolor(bg_rgb)

    # Update suptitle with direct fontsize and font
    title = plt.suptitle("Input Parameter Distributions",
                 fontsize=plotting_style["font"]["sizes"]["main_title"],
                 y=1.00)
    title.set_fontproperties(fonts['regular'])
    
    # Define distributions to plot
    param_info = {
        "Horizon Length at\nSaturation (h_sat)":
            {"key": "h_sat", "unit": "hours", "convert": lambda x: x * 24 * 30},  # Convert months back to hours for display
        "Horizon Doubling\nTime at start (T)": 
            {"key": "horizon_doubling_time", "unit": "months"},
        "Time to\nSaturation (t_sat)":
            {"key": "t_sat", "unit": "days"},
        "Software Progress Rate at\nPresent Day (v_present_day)":
            {"key": "v_present_day", "unit": "x 2024 rate"},
        "Software Progress Rate at\nSaturation (v_software_sat)":
            {"key": "v_software_sat", "unit": "x 2024 rate"},
        "Software Progress Rate at\nSC (v_software_SC)":
            {"key": "v_software_SC", "unit": "x 2024 rate"},
        "Horizon Length at\nSC (h_SC)":
            {"key": "h_SC", "unit": "months"},
        "Delay to become\npublic (d)":
            {"key": "d", "unit": "months"},
    }
    
    # Add algorithmic slowdowns with prettier names
    first_samples = next(iter(all_forecaster_samples.values()))
    for name in first_samples["A_values"].keys():
        param_info[f"Algorithmic Gap\n{get_pretty_name(name)}"] = {
            "key": f"A_values.{name}",
            "unit": "months"  # Already in months
        }
    
    # Plot each distribution
    for (name, info), ax in zip(param_info.items(), axes.flat):
        plot_multi_distribution(ax, name, info, all_forecaster_samples, config, plotting_style, fonts)
    
    # Hide empty subplots
    for ax in axes.flat[len(param_info):]:
        ax.set_visible(False)
    
    # Add legend to first subplot
    axes[0,0].legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    
    # Horizon gap time figure
    fig_derived = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig_derived.add_subplot(111)
    
    # Calculate and plot horizon gap for each forecaster
    stats_text = []
    for forecaster_name, samples in all_forecaster_samples.items():
        # Calculate gap using the same function as the simulation
        g_h, _ = calculate_gaps(samples)
        g_h = g_h/12  # Convert months to years
        
        # Get the base name without any parenthetical text for config lookup
        base_name = forecaster_name.split(" (")[0].lower()
        color = config["forecasters"][base_name]["color"]
        plot_single_distribution(ax, g_h, color=color, label=forecaster_name)
        
        # Calculate and format statistics
        stats = (
            f"{forecaster_name}:\n"
            f"  10th: {np.percentile(g_h, 10):.1f}\n"
            f"  50th: {np.median(g_h):.1f}\n"
            f"  90th: {np.percentile(g_h, 90):.1f}"
        )
        stats_text.append(stats)
    bg_rgb = tuple(int(plotting_style.get("colors", {}).get("background", "#FFFEF8").lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    # Add stats text with white background
    text = ax.text(0.77, 0.98, "\n\n".join(stats_text),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            fontsize=plotting_style["font"]["sizes"]["legend"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Update title with direct fontsize and font
    title = ax.set_title("Horizon Gap Time\nwithout intermediate speedups",
                 fontsize=plotting_style["font"]["sizes"]["title"],
                 pad=10)
    title.set_fontproperties(fonts['regular'])
    
    xlabel = ax.set_xlabel("years",
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    xlabel.set_fontproperties(fonts['regular_small'])
    
    # Set x-axis limit from slightly before 0 to 8 years
    ax.set_xlim(-0.5, 8)
    
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend with specific font
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks with specific font
    ax.tick_params(axis="both", 
                   labelsize=plotting_style["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(fonts['regular_legend'])
    
    return fig_inputs, fig_derived

def plot_multi_distribution(ax: plt.Axes, name: str, info: dict, 
                          all_forecaster_samples: dict, config: dict, plotting_style: dict, fonts: dict) -> None:
    """Helper function to plot distributions from multiple forecasters."""
    background_color = plotting_style.get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Set background color
    ax.set_facecolor(bg_rgb)
    
    # Determine if this should be a log plot
    is_log = name.startswith("Horizon Length at")
    is_t_sat = name.startswith("Time to\nSaturation")
    is_algorithmic_gap = name.startswith("Algorithmic Gap")
    is_non_algorithmic = name.startswith("Non-algorithmic")
    is_speed_param = "Progress" in name
    is_doubling_time = "Doubling" in name
    is_delay = "Delay" in name
    is_h_SC = "SC" in name and "Horizon Length" in name
    
    # Collect all data first for setting appropriate limits
    all_data = []
    for samples in all_forecaster_samples.values():
        if "." in info["key"]:
            key1, key2 = info["key"].split(".")
            data = samples[key1][key2]
        else:
            data = samples[info["key"]]
        
        if "convert" in info:
            data = info["convert"](data)
        all_data.append(data)
    
    all_data = np.concatenate(all_data)
    
    if is_log:
        ax.set_xscale('log')
        # For log plots, use scientific notation
        def format_func(x, p):
            if x == 0:
                return "0"
            exp = int(np.floor(np.log10(x)))
            if exp == 0:
                return "1"
            elif is_h_SC:
                # For h_SC, only show even-numbered powers
                if exp % 2 == 0:
                    return f"$10^{exp}$"
                else:
                    return ""
            else:
                # For other log plots, show all powers
                return f"$10^{exp}$"
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        # Set major ticks at powers of 10 and minor ticks at 2,3,4,5,6,7,8,9
        ax.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=[2,3,4,5,6,7,8,9], numticks=10))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    
    # Get current year for x-axis range if needed
    current_year = datetime.now().year
    
    # Plot distribution for each forecaster
    for forecaster_name, samples in all_forecaster_samples.items():
        # Get the base name without any parenthetical text for config lookup
        base_name = forecaster_name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]
        
        # Get the samples
        if "." in info["key"]:
            key1, key2 = info["key"].split(".")
            data = samples[key1][key2]
        else:
            data = samples[info["key"]]
        
        if "convert" in info:
            data = info["convert"](data)
            
        # For t_sat, convert to actual dates
        if is_t_sat:
            data = data / 12
            current_date = datetime.now()
            current_year = current_date.year
            current_month = current_date.month
            data = current_year + data
        
        # Plot the distribution
        plot_single_distribution(ax, data, color=color, label=forecaster_name)
    
    # Set axis limits based on the type of plot
    if is_t_sat:
        ax.set_xlim(current_year, current_year + 6)
        ax.set_xticks(range(current_year, current_year + 7))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    elif is_log:
        # For log plots, use percentiles that capture the main part of the distribution
        lower, upper = np.percentile(all_data, [0.1, 99.9])
        # Add extra margin in log space
        log_lower, log_upper = np.log10(lower), np.log10(upper)
        margin = (log_upper - log_lower) * 0.1
        ax.set_xlim(10**(log_lower - margin), 10**(log_upper + margin))
    else:
        # For linear plots, use different percentiles based on the type of data
        if is_doubling_time or is_delay:
            # Use tighter bounds for doubling time and delay plots
            lower, upper = np.percentile(all_data, [1, 95])
            margin = (upper - lower) * 0.1
        elif is_algorithmic_gap or is_non_algorithmic:
            lower, upper = np.percentile(all_data, [1, 95])
            margin = (upper - lower) * 0.1
        elif is_speed_param:
            lower, upper = np.percentile(all_data, [1, 97])
            margin = (upper - lower) * 0.15
        else:
            lower, upper = np.percentile(all_data, [0.1, 99.9])
            margin = (upper - lower) * 0.2
        
        ax.set_xlim(max(0, lower - margin), upper + margin)
    
    # Clean up title by removing parenthetical labels
    clean_name = name.split(" (")[0]
    
    # Update subplot titles with direct fontsize and font
    title = ax.set_title(clean_name,
                 fontsize=plotting_style["font"]["sizes"]["title"],
                 pad=10)
    title.set_fontproperties(fonts['regular'])
    
    # Update x-axis label
    if is_t_sat:
        xlabel_text = "Year"
    else:
        # Remove parenthetical unit labels and clean up specific cases
        unit = info["unit"].split(" (")[0]
        if "Rate" in unit and "Progress" in clean_name:
            unit = unit.replace("Rate", "Increase")
        xlabel_text = unit
    
    ax.set_xlabel(xlabel_text,
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.tick_params(axis="both",
                   labelsize=plotting_style["font"]["sizes"]["ticks"])

def plot_single_distribution(ax: plt.Axes, data: np.ndarray, color: str, label: str = None) -> None:
    """Plot a single distribution with kernel density estimation."""
    from scipy.stats import gaussian_kde
    
    # Check if axis is log scale
    is_log = ax.get_xscale() == 'log'
    
    if is_log:
        # For log-scale plots, perform KDE in log space
        log_data = np.log10(data)
        kde = gaussian_kde(log_data)
        
        # Create range in log space with more points near the left tail
        x_min, x_max = np.percentile(log_data, [0.1, 99.9])
        # Add extra margin in log space
        margin = (x_max - x_min) * 0.1
        x_min -= margin
        x_max += margin
        
        # Create non-uniform spacing with more points on the left
        left_range = np.linspace(x_min, x_min + (x_max-x_min)*0.3, 150)  # More points in first 30%
        right_range = np.linspace(x_min + (x_max-x_min)*0.3, x_max, 100)  # Fewer points in the rest
        x_range_log = np.concatenate([left_range, right_range[1:]])  # Remove duplicate point
        
        # Convert back to linear space
        x_range = 10**x_range_log
        
        # Get density in log space and adjust for log scale
        density = kde(x_range_log)
        # This is the key adjustment that was working before:
        density = density / (x_range * np.log(10))
        
        # Normalize density to have similar scale as linear plots
        density = density / np.max(density) * 0.4
    else:
        # For linear plots, use regular KDE
        kde = gaussian_kde(data)
        x_min, x_max = np.percentile(data, [0.1, 99.9])
        margin = (x_max - x_min) * 0.1
        x_range = np.linspace(x_min - margin, x_max + margin, 200)
        density = kde(x_range)
    
    # Plot the density
    ax.plot(x_range, density, '-', color=color, label=label,
            linewidth=2, alpha=0.8, zorder=2)
    ax.fill_between(x_range, density, color=color, alpha=0.1)

def format_year_month(year_decimal: float, max_time: float) -> str:
    """Convert decimal year to Month Year format."""
    if year_decimal >= max_time:
        return f">{int(max_time)}"
        
    year = int(year_decimal)
    month = int((year_decimal % 1) * 12) + 1
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    return f"{month_name} {year}"

def create_research_trajectory_plot(all_forecaster_trajectories: dict, config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create figure showing median research stock trajectory for simulations completing in first half of 2027."""
    background_color = plotting_style.get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Dictionary to store median trajectory data for JSON export
    trajectory_data = {}
    
    # Plot trajectory for each forecaster
    for forecaster_name, (completion_times, research_trajectories, time_trajectories) in all_forecaster_trajectories.items():
        # Filter for simulations that complete in first half of 2027
        filtered_research_trajectories = []
        filtered_time_trajectories = []
        
        for i, completion_time in enumerate(completion_times):
            if 2027.0 <= completion_time < 2027.5:
                # Get the full trajectory for this simulation
                full_research_traj = research_trajectories[i]
                full_time_traj = time_trajectories[i]
                
                filtered_research_trajectories.append(full_research_traj)
                filtered_time_trajectories.append(full_time_traj)
        
        if not filtered_research_trajectories:
            print(f"No simulations for {forecaster_name} completed in first half of 2027")
            continue
            
        print(f"{forecaster_name}: {len(filtered_research_trajectories)} simulations completed in first half of 2027")
        
        # Find the common time grid for interpolation
        # Get the range of all time points
        all_times = []
        for time_traj in filtered_time_trajectories:
            all_times.extend(time_traj)
        
        min_time = min(all_times)
        max_time = max(all_times)
        
        # Create a uniform time grid
        time_grid = np.linspace(min_time, max_time, 500)
        
        # Interpolate each trajectory onto the common time grid
        interpolated_trajectories = []
        for research_traj, time_traj in zip(filtered_research_trajectories, filtered_time_trajectories):
            # Interpolate research stock values onto the time grid
            interp_research = np.interp(time_grid, time_traj, research_traj)
            interpolated_trajectories.append(interp_research)
        
        # Calculate median and percentiles once
        interpolated_trajectories = np.array(interpolated_trajectories)
        median_trajectory = np.median(interpolated_trajectories, axis=0)
        p25_trajectory = np.percentile(interpolated_trajectories, 25, axis=0)
        p75_trajectory = np.percentile(interpolated_trajectories, 75, axis=0)
        
        # Store median trajectory data for JSON export
        trajectory_points = []
        for time_val, median_val, p25_val, p75_val in zip(time_grid, median_trajectory, p25_trajectory, p75_trajectory):
            trajectory_points.append({
                "time": float(time_val),
                "median_research_stock": float(median_val),
                "p25_research_stock": float(p25_val),
                "p75_research_stock": float(p75_val)
            })

        trajectory_data[forecaster_name] = {
            "trajectory": trajectory_points,
            "num_simulations": len(filtered_research_trajectories)
        }

        # Get color from config
        base_name = forecaster_name.split(" (")[0].lower()
        color = config["forecasters"][base_name]["color"]
        
        # Plot median trajectory
        ax.plot(time_grid, median_trajectory, '-', color=color, label=f"{forecaster_name} (median)", 
                linewidth=2.5, alpha=0.9, zorder=2)
        
        # Plot percentiles as shaded area (reusing calculated values)
        ax.fill_between(time_grid, p25_trajectory, p75_trajectory, color=color, alpha=0.2)

    # Save trajectory data to JSON file
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "research_trajectory_data.json", 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    print(f"Saved median trajectory data to {output_dir / 'research_trajectory_data.json'}")
    
    # Configure plot styling
    title = ax.set_title("Research Stock Trajectory\n(Simulations completing Jan-Jun 2027)",
                 fontsize=plotting_style["font"]["sizes"]["title"],
                 pad=15)
    title.set_fontproperties(fonts['regular'])
    
    xlabel = ax.set_xlabel("Year", 
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"],
                 labelpad=10)
    xlabel.set_fontproperties(fonts['regular_small'])
    
    ylabel = ax.set_ylabel("Research Stock", 
                  fontsize=plotting_style["font"]["sizes"]["axis_labels"],
                  labelpad=10)
    ylabel.set_fontproperties(fonts['regular_small'])
    
    # Use scientific notation for y-axis
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Set x-axis to show years
    current_year = datetime.now().year
    x_min = current_year
    x_max = 2028.0  # End at the end of 2027 / beginning of 2028
    ax.set_xlim(x_min, x_max)
    
    # Create custom x-ticks showing quarters or half-years for better resolution
    x_ticks = []
    x_labels = []
    for year in range(current_year, 2029):
        if year <= 2027:
            x_ticks.extend([year, year + 0.5])
            x_labels.extend([str(year), f"{year}.5"])
        else:
            x_ticks.append(year)
            x_labels.append(str(year))
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    
    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both", 
                   labelsize=plotting_style["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontproperties(fonts['regular_legend'])
        if tick in ax.get_xticklabels():
            tick.set_rotation(45)
    
    return fig

def simulate_pre_saturation_period(samples: dict, simulation_idx: int, params: dict, 
                                   forecaster_config: dict, simulation_config: dict) -> tuple[float, float, list[float], list[float]]:
    """Simulate the pre-saturation period with time-based exponential interpolation between progress multipliers at present day and saturation.
        This is a simplification but doesn't affect the results much because the multipliers are low.
    
    Returns:
        tuple: (final_research_stock, final_labor_pool, research_trajectory, time_trajectory)
    """
    t_start = params["t_0"]
    t_sat_years = t_start + samples["t_sat"][simulation_idx]/12  # Convert months to years
    
    if t_sat_years <= t_start:
        # No pre-saturation period needed
        return simulation_config["initial_research_stock"], simulation_config["initial_labor_pool"], [simulation_config["initial_research_stock"]], [t_start]
    
    t = t_start
    current_labor_pool = simulation_config["initial_labor_pool"]
    current_research_stock = simulation_config["initial_research_stock"]
    
    research_trajectory = [current_research_stock]
    time_trajectory = [t]
    
    dt_years = params["dt"] / 365  # Convert days to years
    
    while t < t_sat_years:
        # Simple time-based interpolation fraction
        time_fraction = (t - t_start) / (t_sat_years - t_start)
        time_fraction = np.clip(time_fraction, 0, 1)
        
        # Exponentially interpolate between present day and saturation progress multipliers
        v_present = samples["v_present_day"][simulation_idx]
        v_sat = samples["v_software_sat"][simulation_idx]
        software_prog_multiplier = (1 + v_present) * ((1 + v_sat)/(1 + v_present)) ** time_fraction
        
        # Get current labor growth rate
        current_labor_growth_rate = get_labor_growth_rate(t, forecaster_config["labor_growth_schedule"])
        
        # Convert annual growth rate to daily rate for the time step
        daily_growth_rate = (1 + current_labor_growth_rate) ** (params["dt"]/250) - 1
        
        # Update labor pool
        new_labor = current_labor_pool * daily_growth_rate
        current_labor_pool += new_labor
        
        # Calculate research contribution
        labor_power = simulation_config["labor_power"]
        research_contribution = ((((current_labor_pool+1) ** labor_power)-1) * software_prog_multiplier) / (250/params["dt"])
        
        # Update research stock
        new_research_stock = current_research_stock + research_contribution
        
        # Calculate actual growth rate (annualized)
        actual_growth = (new_research_stock / current_research_stock) ** (250/params["dt"]) - 1
        
        # For the first timestep, set baseline growth
        if len(research_trajectory) == 1:
            baseline_growth = actual_growth
        
        # Calculate adjustment factor based on growth rate ratio
        if baseline_growth > 0:
            growth_ratio = np.log(1 + actual_growth) / np.log(1 + baseline_growth)
        else:
            growth_ratio = 1.0
        
        # Get compute progress rate using compute schedule
        v_compute = get_v_compute(t, forecaster_config["compute_schedule"])
        
        # Get software progress share for this simulation
        software_progress_share = samples["initial_software_progress_share"][simulation_idx]
        
        # Calculate total progress rate using weighted average (though we don't use this for progress tracking in pre-sat)
        v_t = software_progress_share * growth_ratio + (1 - software_progress_share) * v_compute
        
        # Move time forward
        t += dt_years
        
        # Update research stock
        current_research_stock = new_research_stock
        
        # Store trajectory
        research_trajectory.append(current_research_stock)
        time_trajectory.append(t)
        
        # Safety check to prevent infinite loops
        if len(time_trajectory) > 50000:  # Roughly 50k days = ~137 years
            break
    
    return current_research_stock, current_labor_pool, research_trajectory, time_trajectory

def run_and_plot_sc_scenarios(config_path: str = "params.yaml") -> tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure, plt.Figure, dict]:
    """Run SC simulations with different slowdown combinations and plot results for multiple forecasters."""
    print("Loading configuration...")
    # Load configuration
    config = load_config(config_path)
    plotting_style = config["plotting_style"]
    
    # Set up fonts first
    fonts = setup_plotting_style(plotting_style)
    
    # Use t_start from configuration instead of current date
    t_start = config["simulation"]["t_start"]
    
    # Store results for each forecaster
    all_forecaster_headline_results = {}
    all_forecaster_samples = {}
    all_forecaster_trajectories = {}  # Store trajectory data
    
    # Get shared simulation parameters
    sim_params = {
        "n_steps": config["simulation"]["n_steps"],
        "dt": config["simulation"]["dt"],
        "t_0": t_start  # Use t_start from config instead of current date
    }
    
    # Run simulations for each forecaster
    print("\nRunning simulations for each forecaster...")
    for _, forecaster_config in tqdm(config["forecasters"].items(), desc="Forecasters"):
        name = forecaster_config["name"]
        print(f"\nProcessing {name}'s forecasts...")
        
        # Generate samples
        samples = get_distribution_samples(forecaster_config, config["simulation"]["n_sims"], t_start=t_start)
        all_forecaster_samples[name] = samples
        
        # Run simulation
        results, research_trajectories, time_trajectories = run_single_forecaster_simulation(samples, sim_params, forecaster_config, config["simulation"])
        all_forecaster_headline_results[name] = results
        all_forecaster_trajectories[name] = (results, research_trajectories, time_trajectories)
    
    print("\nGenerating plots...")
    # Create plots
    bins = np.arange(2025, 2038, 0.5)
    bin_width = 0.5
    
    fig_headline = create_headline_plot(all_forecaster_headline_results, bins, bin_width, config, plotting_style, fonts)
    fig_inputs, fig_derived = create_distribution_plots(all_forecaster_samples, config, plotting_style, fonts)
    fig_research_trajectory = create_research_trajectory_plot(all_forecaster_trajectories, config, plotting_style, fonts)
    
    # Apply tight layout
    for fig in [fig_headline, fig_inputs, fig_derived, fig_research_trajectory]:
        fig.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving plots...")
    # Save all plots
    fig_headline.savefig(output_dir / "combined_headline.png", dpi=300, bbox_inches="tight")
    fig_inputs.savefig(output_dir / "combined_inputs.png", dpi=300, bbox_inches="tight")
    fig_derived.savefig(output_dir / "combined_derived.png", dpi=300, bbox_inches="tight")
    fig_research_trajectory.savefig(output_dir / "research_trajectory_2027.png", dpi=300, bbox_inches="tight")
    
    # Close all figures to free memory
    plt.close("all")
    
    return fig_headline, None, fig_inputs, fig_derived, fig_research_trajectory, all_forecaster_headline_results

if __name__ == "__main__":
    run_and_plot_sc_scenarios()
    print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
