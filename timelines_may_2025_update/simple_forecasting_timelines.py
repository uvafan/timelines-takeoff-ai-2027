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
    
    # First generate correlated standard normal variables for the two correlated parameters
    n_vars = 2  # horizon_doubling_time, cost_speed
    
    # Create correlation matrix (all pairs have same correlation)
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated standard normal samples
    mean = np.zeros(n_vars)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_sims)
    
    # Convert to uniform using the probability integral transform
    uniform_samples = norm.cdf(normal_samples)
    
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
    
    # Sample horizon length needed for SC (in hours) independently
    dist = get_lognormal_from_80_ci(
        config["distributions"]["h_SC_ci"][0],
        config["distributions"]["h_SC_ci"][1]
    )
    samples["h_SC"] = dist.rvs(n_sims)
    
    # Sample doubling time (in months) with correlation
    dist = get_lognormal_from_80_ci(
        config["distributions"]["horizon_doubling_time_ci"][0],
        config["distributions"]["horizon_doubling_time_ci"][1]
    )
    samples["horizon_doubling_time"] = dist.ppf(uniform_samples[:, 0])
    
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
    
    return samples

def calculate_base_time(samples: dict, current_horizon: float) -> np.ndarray:
    """Calculate base time to reach SC without intermediate speedups."""
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
    
    # Calculate total time based on growth type
    total_time = np.zeros_like(n_doublings)
    
    # For each simulation, calculate time based on growth type and superexponential transition
    for i in range(len(n_doublings)):
        if samples["is_subexponential"][i]:
            # Subexponential case
            growth = samples["sub_doubling_growth_fraction"]
            first_doubling_time = samples["horizon_doubling_time"][i]
            n = n_doublings[i]
            ratio = 1 + growth
            total_time[i] = first_doubling_time * (ratio**n - 1) / (ratio - 1)
        else:
            # Start with exponential growth
            n = n_doublings[i]
            doubling_time = samples["horizon_doubling_time"][i]
            
            # Check if/when it becomes superexponential
            superexponential_start = samples["superexponential_start_time"][i]
            if superexponential_start < np.inf:
                # Calculate how many doublings happen before superexponential transition
                n_before = np.log2(superexponential_start/h_current)
                n_before = min(n_before, n)  # Can't exceed total doublings needed
                
                # Calculate time for exponential phase
                time_before = n_before * doubling_time
                
                # Calculate remaining doublings after transition
                n_after = n - n_before
                if n_after > 0:
                    # Calculate time for superexponential phase
                    decay = samples["se_doubling_decay_fraction"][i]  # Get decay for this specific simulation
                    ratio = 1 - decay
                    time_after = doubling_time * (1 - ratio**n_after) / (1 - ratio)
                    total_time[i] = time_before + time_after
                else:
                    total_time[i] = time_before
            else:
                # Pure exponential case
                total_time[i] = n * doubling_time
    
    # Add cost and speed adjustment
    total_time += samples["cost_speed"]
    
    # Ensure all times are non-negative and finite
    total_time = np.where(np.isfinite(total_time), total_time, 0)
    total_time = np.maximum(total_time, 0)
    
    # Print time distribution by growth type
    print("\nTime Distribution by Growth Type:")
    exp_mask = ~samples["is_subexponential"] & (samples["superexponential_start_time"] == np.inf)
    se_mask = ~samples["is_subexponential"] & (samples["superexponential_start_time"] < np.inf)
    sub_mask = samples["is_subexponential"]
    
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
    
    return total_time

def get_compute_rate(t: float, compute_schedule: list) -> float:
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

def calculate_sc_arrival_year(samples: dict, current_horizon: float, dt: float, human_alg_progress_decrease_date: float, max_simulation_years: float, forecaster_config: dict, simulation_config: dict) -> np.ndarray:
    """Calculate time to reach SC incorporating intermediate speedups and compute scaling."""
    # First calculate base time including cost-and-speed adjustment
    base_time_in_months = calculate_base_time(samples, current_horizon)
    n_sims = len(base_time_in_months)
    
    # Get software progress share from samples
    software_progress_share = samples["initial_software_progress_share"]
    
    # Initialize array for actual times
    ending_times = np.zeros(n_sims)
    
    # Get current date as decimal year
    current_date = datetime.now()
    current_year = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    
    # Convert dt from days to months
    dt_in_months = dt / 30.5
    
    max_time = simulation_config["max_time"]
    
    # Run simulation for each sample with progress bar
    for i in tqdm(range(n_sims), desc="Running simulations", leave=False):
        time = current_year - samples["announcement_delay"][i]/12
        progress = 0.0
        
        # Initialize labor-based research variables
        labor_pool = simulation_config["initial_labor_pool"]
        research_stock = simulation_config["initial_research_stock"]
        labor_power = simulation_config["labor_power"]
        
        # Track previous labor growth rate to detect changes
        prev_labor_growth_rate = None
        
        # Counter for iteration tracking
        iteration_count = 0
        
        while progress < base_time_in_months[i] and time < max_time:
            # Calculate progress fraction
            progress_fraction = progress / base_time_in_months[i]
            
            # Calculate software speedup based on intermediate speedup s(interpolate between present and SC rates)
            software_prog_multiplier = (1 + samples["present_prog_multiplier"][i]) * ((1 + samples["SC_prog_multiplier"][i])/(1 + samples["present_prog_multiplier"][i])) ** progress_fraction

            # Get current labor growth rate from schedule
            current_labor_growth_rate = get_labor_growth_rate(time, forecaster_config["labor_growth_schedule"])
            
            # Convert annual growth rate to daily rate for the time step
            daily_growth_rate = (1 + current_labor_growth_rate) ** (dt/250) - 1

            # Calculate new labor added this period
            new_labor = labor_pool * daily_growth_rate
            labor_pool += new_labor
            
            # Calculate research contribution on a yearly basis, then divide
            research_contribution = ((((labor_pool+1) ** labor_power)-1) * software_prog_multiplier) / (250/dt)

            # Add to research stock
            new_research_stock = research_stock + research_contribution
            
            # Calculate actual growth rate (annualized)
            actual_growth = (new_research_stock / research_stock) ** (250/dt) - 1

            if progress == 0:
                baseline_growth = actual_growth
            
            # Calculate adjustment factor based on growth rate ratio
            # Using log ratio to properly account for compound growth
            growth_ratio = np.log(1 + actual_growth) / np.log(1 + baseline_growth)
            
            # Print growth ratio every 200 iterations
            # if iteration_count % 1000 == 0 and i % 100 == 0:
            #     print(f"\nSimulation {i+1} at {format_year_month(time)}:")
            #     print(f"  Progress: {progress:.2f}/{base_time_in_months[i]:.2f} months ({progress_fraction*100:.1f}%)")
            #     print(f"  Growth ratio: {growth_ratio:.3f}")
            #     print(f"  Labor pool: {labor_pool:.0f}")
            #     print(f"  Research stock: {research_stock:.1f}")

            # Get compute rate for current time using compute schedule
            compute_rate = get_compute_rate(time, forecaster_config["compute_schedule"])
            # Total rate is weighted average of software and compute rates
            total_rate = software_progress_share[i] * growth_ratio + (1 - software_progress_share[i]) * compute_rate
            
            # Update progress and time
            progress += dt_in_months * total_rate
            time += dt_in_months / 12  # Convert months to years
            
            # Update research stock
            research_stock = new_research_stock
            
            # Increment iteration counter
            iteration_count += 1

        # If we hit the time limit, set to max time
        if time >= max_time:
            time = max_time 
            
        ending_times[i] = time
    
    return ending_times

def format_year_month(year_decimal: float, max_time: float = 2050.0) -> str:
    """Convert decimal year to Month Year format."""
    if year_decimal >= max_time:
        return f">{int(max_time)}"
        
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
    current_year = datetime.now().year
    x_min = current_year
    x_max = current_year + 11
    
    # Get max_time from config
    max_time = config["simulation"]["max_time"]
    
    # Plot each forecaster's results
    stats_text = []
    for name, results in all_forecaster_results.items():
        # Get the base name without any parenthetical text for config lookup
        base_name = name.split(" (")[0].lower()
        # Get color from config
        color = config["forecasters"][base_name]["color"]
        
        # Filter out >max_time points for density plot only
        valid_results = [r for r in results if r <= max_time]
        
        # Use KDE for smooth density estimation
        kde = gaussian_kde(valid_results)
        x_range = np.linspace(min(valid_results), max(valid_results), 200)
        density = kde(x_range)
        
        # Plot line with shaded area
        ax.plot(x_range, density, '-', color=color, label=name,
                linewidth=2, alpha=0.8, zorder=2)
        ax.fill_between(x_range, density, color=color, alpha=0.1)
        
        # Calculate statistics using all results to properly show >max_time
        stats = (
            f"{name}:\n"
            f"  10th: {format_year_month(np.percentile(results, 10), max_time)}\n"
            f"  50th: {format_year_month(np.percentile(results, 50), max_time)}\n"
            f"  90th: {format_year_month(np.percentile(results, 90), max_time)}\n"
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
    ax.set_xticks(range(x_min, x_max + 1))
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

def run_simple_sc_simulation(config_path: str = "simple_params.yaml") -> tuple[plt.Figure, dict]:
    """Run simplified SC simulation and plot results."""
    print("Loading configuration...")
    config = load_config(config_path)
    
    # Get current date as decimal year
    current_date = datetime.now()
    current_year_decimal = current_date.year + (current_date.month - 1) / 12 + (current_date.day - 1) / 365.25
    
    # Store results for each forecaster
    all_forecaster_results = {}
    all_forecaster_samples = {}
    
    # Run simulations for each forecaster
    print("\nRunning simulations for each forecaster...")
    with tqdm(total=len(config["forecasters"]), desc="Processing forecasters") as pbar:
        for _, forecaster_config in config["forecasters"].items():
            name = forecaster_config["name"]
            pbar.set_description(f"Processing {name}")
            
            # Generate samples
            samples = get_distribution_samples(forecaster_config, config["simulation"]["n_sims"])
            all_forecaster_samples[name] = samples
            
            # Calculate time to SC
            all_forecaster_results[name] = calculate_sc_arrival_year(
                samples, 
                config["simulation"]["current_horizon"],
                config["simulation"]["dt"],
                config["simulation"]["human_alg_progress_decrease_date"],
                config["simulation"]["max_simulation_years"],
                forecaster_config,
                config["simulation"]
            )
            
            # Print percentage of subexponential simulations
            subexponential_percentage = np.mean(samples["is_subexponential"]) * 100
            print(f"\n{name} subexponential percentage: {subexponential_percentage:.1f}%")
            
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
    
    print("\nGenerating plot...")
    # Create and save plot
    fig = plot_results(all_forecaster_results, config)
    
    # Create output directory if it doesn't exist
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving plot...")
    # Save plot
    fig.savefig(output_dir / "simple_combined_headline.png", dpi=300, bbox_inches="tight")
    
    # Close figure to free memory
    plt.close(fig)
    
    return fig, all_forecaster_results

if __name__ == "__main__":
    run_simple_sc_simulation()
    print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 