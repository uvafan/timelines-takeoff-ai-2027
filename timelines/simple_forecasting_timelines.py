import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, gaussian_kde, norm
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_lognormal_from_80_ci(lower_bound, upper_bound):
    """Generate a lognormal distribution from 80% confidence interval."""
    # Convert to natural log space
    ln_lower = np.log(lower_bound)
    ln_upper = np.log(upper_bound)
    
    # Z-scores for 10th and 90th percentiles
    z_low = -1.28  # norm.ppf(0.1)
    z_high = 1.28  # norm.ppf(0.9)
    
    # Calculate mu and sigma in log space
    sigma = (ln_upper - ln_lower) / (z_high - z_low)
    mu = (ln_upper + ln_lower) / 2
    
    return lognorm(s=sigma, scale=np.exp(mu))

def get_distribution_samples(config: dict, n_sims: int, correlation: float = 0.7) -> dict:
    """Generate samples from all input distributions."""
    samples = {}
    
    # First generate correlated standard normal variables for the three correlated parameters
    n_vars = 4  # T_t, cost_speed, inverse of p_superexponential, present_prog_multiplier, SC_prog_multiplier
    
    # Create correlation matrix (all pairs have same correlation)
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated standard normal samples
    mean = np.zeros(n_vars)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_sims)
    
    # Convert to uniform using the probability integral transform
    uniform_samples = norm.cdf(normal_samples)
    
    # Sample horizon length needed for SC (in hours) independently
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_SC_ci"][0],
        config["distributions"]["h_SC_ci"][1]
    )
    samples["h_SC"] = dist.rvs(n_sims)
    
    # Sample doubling time (in months) with correlation
    dist = get_lognormal_from_80_ci(
        config["distributions"]["T_t_ci"][0],
        config["distributions"]["T_t_ci"][1]
    )
    samples["T_t"] = dist.ppf(uniform_samples[:, 0])
    
    # Sample cost and speed adjustment (in months) with correlation
    dist = get_lognormal_from_80_ci(
        config["distributions"]["cost_speed_ci"][0],
        config["distributions"]["cost_speed_ci"][1]
    )
    samples["cost_speed"] = dist.ppf(uniform_samples[:, 1])
    
    # Sample announcement delay (in months) independently
    dist = get_lognormal_from_80_ci(
        config["distributions"]["announcement_delay_ci"][0],
        config["distributions"]["announcement_delay_ci"][1]
    )
    samples["announcement_delay"] = dist.rvs(n_sims)
    
    # Generate separate correlated samples for progress multipliers
    n_prog_vars = 2  # present_prog_multiplier, SC_prog_multiplier
    prog_corr_matrix = np.array([[1.0, correlation], [correlation, 1.0]])
    prog_normal_samples = np.random.multivariate_normal(np.zeros(n_prog_vars), prog_corr_matrix, size=n_sims)
    prog_uniform_samples = norm.cdf(prog_normal_samples)
    
    # Sample present progress multiplier with correlation to SC multiplier only
    dist = get_lognormal_from_80_ci(
        config["distributions"]["present_prog_multiplier_ci"][0],
        config["distributions"]["present_prog_multiplier_ci"][1]
    )
    samples["present_prog_multiplier"] = dist.ppf(prog_uniform_samples[:, 0])
    
    # Sample SC progress multiplier with correlation to present multiplier only
    dist = get_lognormal_from_80_ci(
        config["distributions"]["SC_prog_multiplier_ci"][0],
        config["distributions"]["SC_prog_multiplier_ci"][1]
    )
    samples["SC_prog_multiplier"] = dist.ppf(prog_uniform_samples[:, 1])
    
    # Sample growth types with correlation for p_superexponential
    p_super = config["distributions"]["p_superexponential"]
    p_sub = config["distributions"]["p_subexponential"]
    
    # Use the correlated uniform sample to determine p_superexponential
    growth_type = uniform_samples[:, 2]
    samples["is_superexponential"] = growth_type < p_super
    samples["is_subexponential"] = (growth_type >= p_super) & (growth_type < (p_super + p_sub))
    samples["is_exponential"] = ~(samples["is_superexponential"] | samples["is_subexponential"])
    
    # Add growth/decay parameters
    samples["se_doubling_decay_fraction"] = config["distributions"]["se_doubling_decay_fraction"]
    samples["sub_doubling_growth_fraction"] = config["distributions"]["sub_doubling_growth_fraction"]
    
    return samples

def calculate_base_time(samples: dict, current_horizon: float) -> tuple[np.ndarray, list]:
    """Calculate base time to reach SC without intermediate speedups and return time-to-horizon mappings."""
    # Convert current horizon from minutes to months
    h_current = current_horizon / (60 * 167)
    
    # Calculate number of doublings needed
    n_doublings = np.log2(samples["h_SC"]/h_current)
    
    # Print distribution statistics for the forecaster
    print("\nDistribution Statistics:")
    
    print(f"\nn_doublings:")
    print(f"  10th percentile: {np.percentile(n_doublings, 10):.2f}")
    print(f"  50th percentile: {np.percentile(n_doublings, 50):.2f}")
    print(f"  90th percentile: {np.percentile(n_doublings, 90):.2f}")
    print(f"  Mean: {np.mean(n_doublings):.2f}")
    print(f"  Std Dev: {np.std(n_doublings):.2f}")
    
    # Print growth type distribution
    exp_mask = samples["is_exponential"]
    se_mask = samples["is_superexponential"]
    sub_mask = samples["is_subexponential"]
    
    n_sims = len(n_doublings)
    total_time = np.zeros(n_sims)
    horizon_mappings = []  # List of time-to-horizon mappings for each simulation
    
    # Use 1 month resolution for efficiency
    dt_mapping = 1.0
    
    # Vectorized calculation of base growth time (before cost_speed)
    growth_time = np.zeros(n_sims)
    
    # For regular exponential cases
    growth_time[exp_mask] = n_doublings[exp_mask] * samples["T_t"][exp_mask]
    
    # For superexponential cases - use analytical formula
    if np.any(se_mask):
        decay = samples["se_doubling_decay_fraction"]
        first_doubling_time = samples["T_t"][se_mask]
        n = n_doublings[se_mask]
        ratio = 1 - decay
        # Sum of geometric series: T1 * (1 - r^n) / (1 - r)
        growth_time[se_mask] = first_doubling_time * (1 - ratio**n) / (1 - ratio)
    
    # For subexponential cases - use analytical formula
    if np.any(sub_mask):
        growth = samples["sub_doubling_growth_fraction"]
        first_doubling_time = samples["T_t"][sub_mask]
        n = n_doublings[sub_mask]
        ratio = 1 + growth
        # Sum of geometric series: T1 * (r^n - 1) / (r - 1)
        growth_time[sub_mask] = first_doubling_time * (ratio**n - 1) / (ratio - 1)
    
    # Total time includes cost_speed adjustment
    total_time = growth_time + samples["cost_speed"]
    
    # Create efficient horizon mappings
    for i in range(n_sims):
        mapping = []
        growth_time_i = growth_time[i]
        cost_speed_time = samples["cost_speed"][i]
        
        # Create time points during growth phase
        if growth_time_i > 0:
            n_growth_points = max(int(growth_time_i / dt_mapping), 2)
            growth_times = np.linspace(0, growth_time_i, n_growth_points)
            
            # Calculate horizon at each time point based on growth type
            if samples["is_exponential"][i]:
                # Exponential: h(t) = h0 * 2^(t/T)
                T_t = samples["T_t"][i]
                horizons = h_current * (2 ** (growth_times / T_t))
                
            elif samples["is_superexponential"][i]:
                # Superexponential: exact analytical formula
                T_t = samples["T_t"][i]
                decay = samples["se_doubling_decay_fraction"]
                horizons = []
                for t in growth_times:
                    if t == 0:
                        horizons.append(h_current)
                    else:
                        # Exact formula: solve for n doublings from t = T_t * (1 - (1-decay)^n) / decay
                        # Rearranging: (1-decay)^n = 1 - t*decay/T_t
                        # So: n = log(1 - t*decay/T_t) / log(1-decay)
                        ratio_term = 1 - t * decay / T_t
                        if ratio_term > 0:
                            n_doublings = np.log(ratio_term) / np.log(1 - decay)
                            horizons.append(h_current * (2 ** n_doublings))
                        else:
                            # If we've exceeded the theoretical limit, use the target
                            horizons.append(samples["h_SC"][i])
                horizons = np.array(horizons)
                
            elif samples["is_subexponential"][i]:
                # Subexponential: exact analytical formula
                T_t = samples["T_t"][i]
                growth = samples["sub_doubling_growth_fraction"]
                horizons = []
                for t in growth_times:
                    if t == 0:
                        horizons.append(h_current)
                    else:
                        # Exact formula: solve for n doublings from t = T_t * ((1+growth)^n - 1) / growth
                        # Rearranging: (1+growth)^n = 1 + t*growth/T_t
                        # So: n = log(1 + t*growth/T_t) / log(1+growth)
                        ratio_term = 1 + t * growth / T_t
                        if ratio_term > 0:
                            n_doublings = np.log(ratio_term) / np.log(1 + growth)
                            horizons.append(h_current * (2 ** n_doublings))
                        else:
                            # Fallback (shouldn't happen for subexponential)
                            horizons.append(samples["h_SC"][i])
                horizons = np.array(horizons)
            
            # Ensure we reach the target horizon
            horizons[-1] = samples["h_SC"][i]
            
            # Convert to minutes and add to mapping
            for t, h in zip(growth_times, horizons):
                mapping.append((t, h * 60 * 167))
        
        # Add cost_speed period (horizon stays constant)
        if cost_speed_time > 0:
            final_horizon = samples["h_SC"][i] * 60 * 167  # Convert to minutes
            cost_speed_times = np.linspace(growth_time_i, growth_time_i + cost_speed_time, 
                                         max(int(cost_speed_time / dt_mapping), 2))
            for t in cost_speed_times:
                mapping.append((t, final_horizon))
        
        horizon_mappings.append(mapping)
    
    # Print time distribution by growth type (using calculated total_time)
    print("\nTime Distribution by Growth Type:")
    for mask, name in [(exp_mask, "Exponential"), (se_mask, "Superexponential"), (sub_mask, "Subexponential")]:
        if np.any(mask):
            times = total_time[mask]
            print(f"\n{name}:")
            print(f"  10th percentile: {np.percentile(times, 10):.2f} months")
            print(f"  50th percentile: {np.percentile(times, 50):.2f} months")
            print(f"  90th percentile: {np.percentile(times, 90):.2f} months")
            print(f"  Mean: {np.mean(times):.2f} months")
            print(f"  Std Dev: {np.std(times):.2f} months")
    
    # Print overall time distribution
    print("\nOverall Time Distribution:")
    print(f"  10th percentile: {np.percentile(total_time, 10):.2f} months")
    print(f"  50th percentile: {np.percentile(total_time, 50):.2f} months")
    print(f"  90th percentile: {np.percentile(total_time, 90):.2f} months")
    print(f"  Mean: {np.mean(total_time):.2f} months")
    print(f"  Std Dev: {np.std(total_time):.2f} months")
    
    return total_time, horizon_mappings

def get_compute_rate(t: float, compute_decrease_date: float) -> float:
    """Calculate compute progress rate based on time."""
    return 0.5 if t >= compute_decrease_date else 1.0

def calculate_sc_arrival_year(samples: dict, current_horizon: float, dt: float, compute_decrease_date: float, human_alg_progress_decrease_date: float, max_simulation_years: float) -> np.ndarray:
    """Calculate time to reach SC incorporating intermediate speedups and compute scaling."""
    # First calculate base time including cost-and-speed adjustment
    base_time_in_months, _ = calculate_base_time(samples, current_horizon)
    n_sims = len(base_time_in_months)
    
    # Initialize array for actual times
    ending_times = np.zeros(n_sims)
    
    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year = 2025.25
    
    # Convert dt from days to months
    dt = dt / 30.5
    
    max_time = 2050
    
    # Run simulation for each sample with progress bar
    for i in tqdm(range(n_sims), desc="Running simulations", leave=False):
        time = current_year - samples["announcement_delay"][i]/12
    
        progress = 0.0
        
        while progress < base_time_in_months[i] and time < max_time:
            # Calculate progress fraction
            progress_fraction = progress / base_time_in_months[i]
            
            # Calculate algorithmic speedup based on intermediate speedup s(interpolate between present and SC rates)
            v_algorithmic = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** progress_fraction

            # adjust algorithmic rate if human alg progress has decreased, in betweene
            if time >= human_alg_progress_decrease_date:
                only_multiplier = v_algorithmic * 0.5
                only_additive = v_algorithmic - 0.5
                # geometric mean of only_multiplier and only_additive, aggregating between extremes of how AIs/humans could complement
                v_algorithmic = np.sqrt(only_multiplier * only_additive)
            
            # Get compute rate for current time (not affected by intermediate speedups)
            compute_rate = get_compute_rate(time, compute_decrease_date)
            # Total rate is mean of algorithmic and compute rates
            total_rate = (v_algorithmic + compute_rate) / 2
            
            # Update progress and time
            progress += dt * total_rate
            time += dt / 12  # Convert months to years
        
        # If we hit the time limit, set to max time
        if time >= max_time:
            time = max_time 
            
        ending_times[i] = time
    
    return ending_times


def calculate_sc_arrival_year_with_trajectories(samples: dict, current_horizon: float, dt: float, compute_decrease_date: float, human_alg_progress_decrease_date: float, max_simulation_years: float) -> tuple[np.ndarray, list]:
    """Calculate time to reach SC incorporating intermediate speedups and compute scaling, returning both ending times and trajectories."""
    # First calculate base time including cost-and-speed adjustment and get horizon mappings
    base_time_in_months, horizon_mappings = calculate_base_time(samples, current_horizon)
    n_sims = len(base_time_in_months)
    
    # Initialize array for actual times
    ending_times = np.zeros(n_sims)
    
    # Store trajectories for each simulation
    trajectories = []
    
    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year = 2025.25
    
    # Convert dt from days to months
    dt = dt / 30.5
    
    max_time = 2050
    
    # Helper function to interpolate horizon from mapping
    def get_horizon_at_progress(mapping, progress_months):
        """Get horizon at given progress using linear interpolation."""
        if not mapping:
            return current_horizon
        
        # Find the appropriate time point in the mapping
        times = [t for t, h in mapping]
        horizons = [h for t, h in mapping]
        
        if progress_months <= times[0]:
            return horizons[0]
        elif progress_months >= times[-1]:
            return horizons[-1]
        else:
            # Linear interpolation
            for i in range(len(times) - 1):
                if times[i] <= progress_months <= times[i + 1]:
                    t1, t2 = times[i], times[i + 1]
                    h1, h2 = horizons[i], horizons[i + 1]
                    # Linear interpolation
                    ratio = (progress_months - t1) / (t2 - t1)
                    return h1 + ratio * (h2 - h1)
        
        return horizons[-1]  # Fallback
    
    # Run simulation for each sample with progress bar
    for i in tqdm(range(n_sims), desc="Running simulations", leave=False):
        time = current_year - samples["announcement_delay"][i]/12
    
        progress = 0.0
        
        # Store trajectory for this simulation
        trajectory = []
        
        while progress < base_time_in_months[i] and time < max_time:
            # Calculate progress fraction
            progress_fraction = progress / base_time_in_months[i]
            
            # Get current horizon from the mapping
            current_horizon_minutes = get_horizon_at_progress(horizon_mappings[i], progress)
            
            # Store trajectory point (time, horizon in minutes)
            trajectory.append((time+samples["announcement_delay"][i]/12, current_horizon_minutes))
            
            # Calculate algorithmic speedup based on intermediate speedup s(interpolate between present and SC rates)
            v_algorithmic = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** progress_fraction

            # adjust algorithmic rate if human alg progress has decreased, in between
            if time >= human_alg_progress_decrease_date:
                only_multiplier = v_algorithmic * 0.5
                only_additive = v_algorithmic - 0.5
                # geometric mean of only_multiplier and only_additive, aggregating between extremes of how AIs/humans could complement
                v_algorithmic = np.sqrt(only_multiplier * only_additive)
            
            # Get compute rate for current time (not affected by intermediate speedups)
            compute_rate = get_compute_rate(time, compute_decrease_date)
            # Total rate is mean of algorithmic and compute rates
            total_rate = (v_algorithmic + compute_rate) / 2
            
            # Update progress and time
            progress += dt * total_rate
            time += dt / 12  # Convert months to years
        
        # If we hit the time limit, set to max time
        if time >= max_time:
            time = max_time 
            
        ending_times[i] = time
        trajectories.append(trajectory)
    
    return ending_times, trajectories

def format_year_month(year_decimal: float) -> str:
    """Convert decimal year to Month Year format."""
    if year_decimal >= 2050:
        return ">2050"
        
    year = int(year_decimal)
    month = int((year_decimal % 1) * 12) + 1
    month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
    return f"{month_name} {year}"

def plot_results(all_forecaster_results: dict, config: dict) -> plt.Figure:
    """Create plot showing results from all forecasters."""
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Set monospace font
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family
    
    fig = plt.figure(figsize=(10, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Get current year for x-axis range
    current_year = 2025.25
    # current_year = datetime.now().year
    x_min = current_year
    x_max = current_year + 11
    
    # Plot each forecaster's results
    stats_text = []
    for name, results in all_forecaster_results.items():
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]
        
        # Filter out >2050 points for density plot only
        valid_results = [r for r in results if r <= 2050]
        
        # Use KDE for smooth density estimation
        kde = gaussian_kde(valid_results)
        x_range = np.linspace(min(valid_results), max(valid_results), 200)
        density = kde(x_range)
        
        # Plot line with shaded area
        ax.plot(x_range, density, '-', color=color, label=name,
                linewidth=2, alpha=0.8, zorder=2)
        ax.fill_between(x_range, density, color=color, alpha=0.1)
        
        # Calculate statistics using all results to properly show >2050
        stats = (
            f"{name}:\n"
            f"  10th: {format_year_month(np.percentile(results, 10))}\n"
            f"  50th: {format_year_month(np.percentile(results, 50))}\n"
            f"  90th: {format_year_month(np.percentile(results, 90))}\n"
        )
        stats_text.append(stats)

    
    # Add statistics text box
    ax.text(0.7, 0.95, "\n\n".join(stats_text),
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            # bbox=dict(facecolor=bg_rgb, alpha=0.9,
            #          edgecolor=config["plotting_style"]["colors"]["human"]["dark"],
            #          linewidth=0.5),
            fontsize=config["plotting_style"]["font"]["sizes"]["legend"])
    
    # Configure plot
    ax.set_title("Superhuman Coder Arrival, Time Horizon Extension",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Probability Density", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    
    # Set axis properties
    ax.set_xticks(range(int(x_min), int(x_max) + 1))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, None)
    
    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=config["plotting_style"]["font"]["sizes"]["legend"])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    return fig

def plot_march_2027_trajectories(all_forecaster_results: dict, all_forecaster_trajectories: dict, all_forecaster_samples: dict, config: dict) -> plt.Figure:
    """Create plot showing time horizon trajectories for runs that reach SC in March 2027."""
    background_color = config["plotting_style"].get("colors", {}).get("background", "#FFFEF8")
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    # Set monospace font
    font_family = config["plotting_style"].get("font", {}).get("family", "monospace")
    plt.rcParams['font.family'] = font_family
    
    fig = plt.figure(figsize=(12, 8), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Define March 2027 as decimal year (2027.167)
    march_2027 = 2027 + (3 - 1) / 12  # March is month 3
    tolerance = 0.1  # Within 1.2 months after March 2027
    
    # Get current year for x-axis range
    # current_year = datetime.now().year
    current_year = 2025.25
    x_min = current_year
    x_max = 2028
    
    total_trajectories_plotted = 0
    all_final_horizons = []  # Collect final horizon times across all forecasters (March 2027 only)
    all_final_horizons_all_runs = []  # Collect final horizon times for ALL runs
    all_h_sc_samples = []  # Collect h_SC samples across all forecasters
    
    # Plot trajectories for each forecaster
    for name, results in all_forecaster_results.items():
        trajectories = all_forecaster_trajectories[name]
        
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]
        
        # Find runs that reach SC in March 2027 (within tolerance)
        march_2027_runs = []
        final_horizons_this_forecaster = []
        final_horizons_all_runs_this_forecaster = []
        
        # First, collect ALL final horizon times for this forecaster
        for i, trajectory in enumerate(trajectories):
            if trajectory:  # Make sure trajectory is not empty
                final_horizon_minutes = trajectory[-1][1]  # Last horizon value
                final_horizons_all_runs_this_forecaster.append(final_horizon_minutes)
                all_final_horizons_all_runs.append(final_horizon_minutes)
        
        # Then find March 2027 runs specifically
        for i, end_time in enumerate(results):
            if end_time - march_2027 > 0 and end_time - march_2027 <= tolerance:
                march_2027_runs.append(i)
                # Get the final horizon time from the trajectory
                trajectory = trajectories[i]
                if trajectory:  # Make sure trajectory is not empty
                    final_horizon_minutes = trajectory[-1][1]  # Last horizon value
                    final_horizons_this_forecaster.append(final_horizon_minutes)
                    all_final_horizons.append(final_horizon_minutes)
        
        print(f"{name}: Found {len(march_2027_runs)} runs reaching SC in March 2027")
        
        # Print h_SC distribution for this forecaster
        h_sc_samples = all_forecaster_samples[name]["h_SC"]
        all_h_sc_samples.extend(h_sc_samples)  # Collect for overall distribution
        h_sc_work_months = h_sc_samples  # Convert from minutes to work months
        print(f"  h_SC target distribution (work months):")
        print(f"    10th percentile: {np.percentile(h_sc_work_months, 10):,.2f}")
        print(f"    50th percentile: {np.percentile(h_sc_work_months, 50):,.2f}")
        print(f"    90th percentile: {np.percentile(h_sc_work_months, 90):,.2f}")
        print(f"    Mean: {np.mean(h_sc_work_months):,.2f}")
        
        # Print horizon distribution for ALL runs for this forecaster
        if final_horizons_all_runs_this_forecaster:
            horizons_all_array = np.array(final_horizons_all_runs_this_forecaster)
            print(f"  ALL RUNS final horizon distribution (work months):")
            horizons_all_work_months = horizons_all_array / (60 * 167)
            print(f"    10th percentile: {np.percentile(horizons_all_work_months, 10):,.2f}")
            print(f"    50th percentile: {np.percentile(horizons_all_work_months, 50):,.2f}")
            print(f"    90th percentile: {np.percentile(horizons_all_work_months, 90):,.2f}")
            print(f"    Mean: {np.mean(horizons_all_work_months):,.2f}")
        
        # Print horizon distribution for March 2027 runs for this forecaster
        if final_horizons_this_forecaster:
            horizons_array = np.array(final_horizons_this_forecaster)
            # Convert to work months (167 hours per work month)
            horizons_work_months = horizons_array / (60 * 167)
            print(f"  March 2027 final horizon distribution (work months):")
            print(f"    10th percentile: {np.percentile(horizons_work_months, 10):,.2f}")
            print(f"    50th percentile: {np.percentile(horizons_work_months, 50):,.2f}")
            print(f"    90th percentile: {np.percentile(horizons_work_months, 90):,.2f}")
            print(f"    Mean: {np.mean(horizons_work_months):,.2f}")
            print()
        
        # Plot trajectories for these runs
        for run_idx in march_2027_runs:
            trajectory = trajectories[run_idx]
            if trajectory:  # Make sure trajectory is not empty
                times, horizons = zip(*trajectory)
                ax.plot(times, horizons, '-', color=color, alpha=0.3, linewidth=1)
                total_trajectories_plotted += 1
    
    print(f"\nTotal trajectories plotted: {total_trajectories_plotted}")
    
    # Print overall h_SC distribution across all forecasters
    if all_h_sc_samples:
        print(f"\nOVERALL h_SC Target Distribution:")
        print(f"Total samples: {len(all_h_sc_samples)}")
        all_h_sc_array = np.array(all_h_sc_samples)
        all_h_sc_work_months = all_h_sc_array
        print(f"h_SC target distribution (work months):")
        print(f"  10th percentile: {np.percentile(all_h_sc_work_months, 10):,.2f}")
        print(f"  50th percentile: {np.percentile(all_h_sc_work_months, 50):,.2f}")
        print(f"  90th percentile: {np.percentile(all_h_sc_work_months, 90):,.2f}")
        print(f"  Mean: {np.mean(all_h_sc_work_months):,.2f}")
        print()
    
    # Print overall horizon distribution for ALL runs across all forecasters
    if all_final_horizons_all_runs:
        print(f"OVERALL Final Horizon Distribution for ALL RUNS:")
        print(f"Total samples: {len(all_final_horizons_all_runs)}")
        all_horizons_all_array = np.array(all_final_horizons_all_runs)
        
        # Convert to work months (167 hours per work month)
        all_horizons_all_work_months = all_horizons_all_array / (60 * 167)
        print(f"Final horizon distribution (work months):")
        print(f"  10th percentile: {np.percentile(all_horizons_all_work_months, 10):,.2f}")
        print(f"  50th percentile: {np.percentile(all_horizons_all_work_months, 50):,.2f}")
        print(f"  90th percentile: {np.percentile(all_horizons_all_work_months, 90):,.2f}")
        print(f"  Mean: {np.mean(all_horizons_all_work_months):,.2f}")
        print()
    
    # Print overall horizon distribution across all forecasters for March 2027 runs
    if all_final_horizons:
        print(f"OVERALL Final Horizon Distribution for March 2027 SC arrivals:")
        print(f"Total samples: {len(all_final_horizons)}")
        all_horizons_array = np.array(all_final_horizons)
        
        # Convert to work months (167 hours per work month)
        all_horizons_work_months = all_horizons_array / (60 * 167)
        print(f"Final horizon distribution (work months):")
        print(f"  10th percentile: {np.percentile(all_horizons_work_months, 10):,.2f}")
        print(f"  50th percentile: {np.percentile(all_horizons_work_months, 50):,.2f}")
        print(f"  90th percentile: {np.percentile(all_horizons_work_months, 90):,.2f}")
        print(f"  Mean: {np.mean(all_horizons_work_months):,.2f}")
        print()
    
    # Add reference trajectory line with specific points
    reference_times = [2025.25, 2026, 2026.5, 2027]
    reference_horizons = [
        15,  # 15 minutes
        240,  # 4 work hours = 4 * 60 minutes
        4800,  # 2 work weeks = 2 * 40 hours * 60 minutes
        320640  # 32 work months = 32 * 167 hours * 60 minutes
    ]
    ax.plot(reference_times, reference_horizons, 'o-', color='purple', 
            linewidth=3, markersize=6, alpha=0.8, 
            label='Reference Timeline', zorder=10)
    
    # Add horizontal line for SC threshold (assuming it's the target horizon)
    # We'll use the current horizon as a reference point
    current_horizon = config["simulation"]["current_horizon"]
    ax.axhline(y=current_horizon, color='red', linestyle='--', alpha=0.7, 
               label='Current Horizon (15 min)', linewidth=2)
    
    # Add vertical line for March 2027
    ax.axvline(x=march_2027, color='blue', linestyle='--', alpha=0.7, 
               label='March 2027', linewidth=2)
    
    # Configure plot
    ax.set_title("Time Horizon Trajectories for Runs Reaching SC in March 2027",
                 fontsize=config["plotting_style"]["font"]["sizes"]["title"])
    ax.set_xlabel("Year", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Time Horizon", fontsize=config["plotting_style"]["font"]["sizes"]["axis_labels"])
    
    # Set axis properties
    ax.set_xlim(x_min, x_max)
    ax.set_yscale('log')  # Log scale for time horizon
    
    # Create dynamic y-axis labels with human-readable time units
    def format_time_label(minutes):
        """Convert minutes to human-readable time labels."""
        if minutes < 1:
            return f"{minutes*60:.0f}s"
        elif minutes < 60:
            return f"{minutes:.0f}m"
        elif minutes < 1440:  # 24 hours
            hours = minutes / 60
            return f"{hours:.0f}h"
        elif minutes < 10080:  # 1 week
            days = minutes / 1440
            return f"{days:.0f}d"
        elif minutes < 43200:  # 1 month (30 days)
            weeks = minutes / 10080
            return f"{weeks:.0f}w"
        elif minutes < 525600:  # 1 year
            months = minutes / 43200
            return f"{months:.0f}mo"
        else:
            years = minutes / 525600
            return f"{years:.0f}y"
    
    # Get current y-axis limits to determine appropriate tick positions
    ax.autoscale_view()
    y_min, y_max = ax.get_ylim()
    
    # Generate tick positions that make sense for time horizons
    # Use a more intelligent spacing to avoid crowding
    tick_positions = []
    tick_labels = []
    
    # Define all possible tick positions with their labels
    all_ticks = [
        # Short times
        (0.1, "6s"),
        (0.5, "30s"),
        (1, "1 min"),
        (5, "5 min"),
        (15, "15 min"),
        (30, "30 min"),
        (60, "1 hour"),
        (240, "4 hours"),
        # Work-based time units
        (480, "1 work day"),      # 8 hours
        (2400, "1 work week"),    # 40 hours  
        (10020, "1 work month"),  # 167 hours
        (60120, "6 work months"), # 6 * 167 hours
        (120240, "1 work year"),  # 12 * 167 hours
        (240480, "2 work years"), # 24 * 167 hours
        (601200, "5 work years"), # 60 * 167 hours
        (1202400, "10 work years"), # 120 * 167 hours
        (2404800, "20 work years"), # 240 * 167 hours
        (4809600, "40 work years"), # 480 * 167 hours
    ]
    
    # Filter ticks to be within range and well-spaced
    valid_ticks = [(pos, label) for pos, label in all_ticks if y_min <= pos <= y_max]
    
    # If we have too many ticks, thin them out intelligently
    if len(valid_ticks) > 8:
        # Keep every other tick, but always keep the first and last
        filtered_ticks = [valid_ticks[0]]  # Always keep first
        for i in range(2, len(valid_ticks) - 1, 2):  # Take every other middle tick
            filtered_ticks.append(valid_ticks[i])
        if len(valid_ticks) > 1:
            filtered_ticks.append(valid_ticks[-1])  # Always keep last
        valid_ticks = filtered_ticks
    
    # Set the final ticks
    if valid_ticks:
        tick_positions, tick_labels = zip(*valid_ticks)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    
    # Grid and spines
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    ax.legend(fontsize=config["plotting_style"]["font"]["sizes"]["legend"])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=config["plotting_style"]["font"]["sizes"]["ticks"])
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    
    return fig

def run_simple_sc_simulation(config_path: str = "simple_params.yaml", use_step_simulation: bool = False) -> tuple[plt.Figure, dict]:
    """Run simplified SC simulation and plot results."""
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Get current date as decimal year
    # current_date = datetime.now()
    # current_year_decimal = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    current_year_decimal = 2025.25
    
    # Store results for each forecaster
    all_forecaster_results = {}
    all_forecaster_samples = {}
    all_forecaster_trajectories = {}
    
    # Run simulations for each forecaster
    print("\nRunning simulations for each forecaster...")
    with tqdm(total=len(config["forecasters"]), desc="Processing forecasters") as pbar:
        for _, forecaster_config in config["forecasters"].items():
            name = forecaster_config["name"]
            pbar.set_description(f"Processing {name}")
            
            # Generate samples
            samples = get_distribution_samples(forecaster_config, config["simulation"]["n_sims"])
            all_forecaster_samples[name] = samples
            
            # Calculate time to SC with trajectories
            results, trajectories = calculate_sc_arrival_year_with_trajectories(
                samples, 
                config["simulation"]["current_horizon"],
                config["simulation"]["dt"],
                config["simulation"]["compute_decrease_date"],
                config["simulation"]["human_alg_progress_decrease_date"],
                config["simulation"]["max_simulation_years"]
            )
            
            all_forecaster_results[name] = results
            all_forecaster_trajectories[name] = trajectories
            
            pbar.update(1)
    
    # Print debug information grouped by parameter
    # print("\nDebug Information by Parameter:")
    # print("=" * 80)
    
    # Define parameters to analyze
    # parameters = {
    #     "h_SC": "Time Horizon for SC (months)",
    #     "T_t": "Doubling Time (months)",
    #     "cost_speed": "Cost and Speed Adjustment (months)",
    #     "announcement_delay": "Announcement Delay (months)",
    # }
    
    # for param, description in parameters.items():
    #     print(f"\n{description}:")
    #     print("-" * 40)
    #     for name, samples in all_forecaster_samples.items():
    #         if param.startswith("is_"):
    #             # For boolean parameters, calculate percentage
    #             value = np.mean(samples[param]) * 100
    #             print(f"{name:>10}: {value:>6.1f}%")
    #         else:
    #             # For numeric parameters, show percentiles
    #             data = samples[param]
    #             print(f"{name:>10}:")
    #             print(f"          10th: {np.percentile(data, 10):>6.2f}")
    #             print(f"          50th: {np.percentile(data, 50):>6.2f}")
    #             print(f"          90th: {np.percentile(data, 90):>6.2f}")
    
    print("\nGenerating plots...")
    # Create and save original plot
    fig = plot_results(all_forecaster_results, config)
    
    # Create and save trajectory plot
    fig_trajectories = plot_march_2027_trajectories(all_forecaster_results, all_forecaster_trajectories, all_forecaster_samples, config)
    
    # Create output directory if it doesn't exist
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving plots...")
    # Save plots
    fig.savefig(output_dir / "simple_combined_headline.png", dpi=300, bbox_inches="tight")
    fig_trajectories.savefig(output_dir / "march_2027_trajectories.png", dpi=300, bbox_inches="tight")
    
    # Close figures to free memory
    plt.close(fig)
    plt.close(fig_trajectories)
    
    return fig, all_forecaster_results

if __name__ == "__main__":
    # Run with closed-form solution (faster)
    print("=== Running with closed-form solution ===")
    run_simple_sc_simulation(use_step_simulation=False)
    
    # Uncomment below to also run with step-by-step simulation (slower but more accurate)
    # print("\n=== Running with step-by-step simulation ===")
    # run_simple_sc_simulation(use_step_simulation=True)
    
    print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 