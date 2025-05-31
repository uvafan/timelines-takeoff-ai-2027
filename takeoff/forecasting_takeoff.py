import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import yaml
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_lognormal_from_80_ci(lower_bound, upper_bound):
    """Get lognormal distribution parameters from 80% confidence interval."""
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

def get_project_progress_samples(config: dict, n_sims: int) -> dict:
    """Generate progress rate samples for each project using lognormal distributions with ceilings."""
    project_samples = {}
    
    for project_name, params in config["projects"].items():
        # Check if using new lognormal format or old fixed format
        if isinstance(params, dict) and "lower_bound" in params:
            # New lognormal format
            lower_bound = params["lower_bound"]
            upper_bound = params["upper_bound"]
            ceiling = params["ceiling"]
            
            # Generate lognormal samples
            dist = get_lognormal_from_80_ci(lower_bound, upper_bound)
            samples = dist.rvs(n_sims)
            
            # Apply ceiling constraint
            samples = np.minimum(samples, ceiling)
            
            project_samples[project_name] = samples
        else:
            # Old fixed format - create array of identical values
            fixed_rate = params if isinstance(params, (int, float)) else 1.0
            project_samples[project_name] = np.full(n_sims, fixed_rate)
    
    return project_samples

def get_project_starting_positions(config: dict) -> dict:
    """Extract starting positions for each project from config."""
    starting_positions = {}
    
    for project_name, params in config["projects"].items():
        if isinstance(params, dict) and "starting_position" in params:
            starting_positions[project_name] = params["starting_position"]
        else:
            starting_positions[project_name] = 0.0  # Default to starting at beginning
    
    return starting_positions

def get_milestone_samples(config: dict, n_sims: int, correlation: float = 0.7) -> dict:
    """Generate samples for milestone timings and speeds with correlation between gap sizes."""
    samples = {}
    
    # Parse starting time
    start_date = datetime.strptime(config["starting_time"], "%B %d %Y")
    samples["start_time"] = start_date
    
    # Get list of time gaps to model
    milestone_pairs = list(config["times"].keys())
    n_vars = len(milestone_pairs)
    
    # Create correlation matrix (all pairs have same correlation)
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    
    # Generate correlated standard normal samples
    mean = np.zeros(n_vars)
    normal_samples = np.random.multivariate_normal(mean, corr_matrix, size=n_sims)
    
    # Convert to uniform using the probability integral transform
    uniform_samples = norm.cdf(normal_samples)
    
    # Generate AMR to SAR samples with correlation to SC to SAR
    amr_sar_sc_sar_correlation = 0.8
    corr_matrix_2 = np.array([[1.0, amr_sar_sc_sar_correlation],
                             [amr_sar_sc_sar_correlation, 1.0]])
    mean_2 = np.zeros(2)
    correlated_normal_samples = np.random.multivariate_normal(mean_2, corr_matrix_2, size=n_sims)
    uniform_samples_corr = norm.cdf(correlated_normal_samples)
    
    # Generate AMR to SAR samples using the correlated uniforms
    amr_to_sar_dist = get_lognormal_from_80_ci(1, 25)
    amr_to_sar_samples = amr_to_sar_dist.ppf(uniform_samples_corr[:, 0])
    
    # Generate SAR to SIAR equivalent jumps samples
    sar_to_siar_equiv_jumps_dist = get_lognormal_from_80_ci(0.3, 7.5)
    sar_to_siar_equiv_jumps_samples = sar_to_siar_equiv_jumps_dist.ppf(np.random.random(n_sims))
    
    # Generate time gap samples
    samples["time_gaps"] = {}
    
    for idx, milestone_pair in enumerate(milestone_pairs):
        params = config["times"][milestone_pair]
        p_zero, lower, upper = params
        
        if milestone_pair == "SAR to SIAR":
            # Calculate SAR to SIAR time using the formula
            values = np.zeros(n_sims)
            for i in range(n_sims):
                x = 10 + amr_to_sar_samples[i]
                values[i] = (x * 2**(2*np.log(x/10)/np.log(2)))-x
            
            # Print distribution statistics for SAR to SIAR
            print("\nSAR to SIAR human-only time distribution (years):")
            print(f"10th percentile: {np.percentile(values, 10):.2f}")
            print(f"25th percentile: {np.percentile(values, 25):.2f}")
            print(f"50th percentile (median): {np.percentile(values, 50):.2f}")
            print(f"75th percentile: {np.percentile(values, 75):.2f}")
            print(f"90th percentile: {np.percentile(values, 90):.2f}")
            print(f"Min: {np.min(values):.2f}")
            print(f"Max: {np.max(values):.2f}")
        
        elif milestone_pair == "SIAR to ASI":
            # Calculate SIAR to ASI time using the new formula
            values = np.zeros(n_sims)
            for i in range(n_sims):
                stock_through_sar = 10 + amr_to_sar_samples[i]
                sar_to_siar_years = samples["time_gaps"]["SAR to SIAR"][i] / 365  # Convert back to years
                sar_to_siar_equiv_jumps = sar_to_siar_equiv_jumps_samples[i]
                
                x = stock_through_sar + sar_to_siar_years
                values[i] = (x * 2**(sar_to_siar_equiv_jumps*np.log(x/stock_through_sar)/np.log(2)))-x
            
            # Print distribution statistics for SIAR to ASI
            print("\nSIAR to ASI human-only time distribution (years):")
            print(f"10th percentile: {np.percentile(values, 10):.2f}")
            print(f"25th percentile: {np.percentile(values, 25):.2f}")
            print(f"50th percentile (median): {np.percentile(values, 50):.2f}")
            print(f"75th percentile: {np.percentile(values, 75):.2f}")
            print(f"90th percentile: {np.percentile(values, 90):.2f}")
            print(f"Min: {np.min(values):.2f}")
            print(f"Max: {np.max(values):.2f}")
        
        elif milestone_pair == "SC to SAR":
            # Generate lognormal samples using the correlated uniform samples
            dist = get_lognormal_from_80_ci(lower, upper)
            values = dist.ppf(uniform_samples_corr[:, 1])
            is_nonzero = np.random.random(n_sims) >= p_zero
            values[~is_nonzero] = 0
            
        elif milestone_pair == "PRESENT_DAY to SC":
            # Fixed 5-year gap (the baseline journey to SC)
            values = np.full(n_sims, 5.0)  # Always 5 years from present day to SC
            
        else:
            # Standard handling for other transitions
            is_nonzero = np.random.random(n_sims) >= p_zero
            dist = get_lognormal_from_80_ci(lower, upper)
            values = dist.ppf(uniform_samples[:, idx])
            values[~is_nonzero] = 0
        
        # Convert years to days
        samples["time_gaps"][milestone_pair] = values * 365
    
    # Store speedup values
    samples["speeds"] = {}
    
    # Store fixed speedup values for other milestones
    for milestone, speed in config["speedups"].items():
        samples["speeds"][milestone] = speed
    
    return samples

def run_phase_simulation(gap: float, start_speed: float, end_speed: float, progress_rate: float = 1.0, milestone_pair: str = None) -> float:
    """Run simulation for a single phase with exponential speedup.
    
    Args:
        gap: Required progress in days
        start_speed: Initial speed multiplier v for this phase
        end_speed: Final speed multiplier v for this phase
        progress_rate: Rate at which this actor makes progress (1.0 = normal, 0.3 = 30% speed)
        milestone_pair: String identifying which transition this is
        
    Returns:
        Calendar days taken to complete the phase
    """
    dt = 1  # One day timesteps
    calendar_time = 0
    progress = 0
    
    # Cap to prevent overflow
    MAX_CALENDAR_DAYS = 365 * 1000
    
    while progress < gap:
        # Calculate current speedup based on progress through the phase
        progress_ratio = progress / gap
        current_speedup = start_speed * (end_speed/start_speed)**progress_ratio
        
        # Make progress at varying speed, adjusted by progress_rate
        progress += current_speedup * progress_rate * dt
        calendar_time += dt
        
        if calendar_time > MAX_CALENDAR_DAYS:
            print(f"Warning: Phase duration capped at {MAX_CALENDAR_DAYS/365:.1f} years")
            return MAX_CALENDAR_DAYS
    
    return calendar_time

def run_single_simulation_with_tracking(samples: dict, sim_idx: int, progress_rate: float = 1.0, starting_position: float = 0.0) -> tuple[list[datetime], list[float]]:
    """Run a single simulation and track both milestone dates and phase durations.
    
    Args:
        samples: Dictionary containing milestone samples
        sim_idx: Simulation index
        progress_rate: Rate at which this actor makes progress (1.0 = normal)
        starting_position: How many years of progress toward SC have already been completed
    """
    milestone_dates = []
    phase_calendar_days = []
    current_date = samples["start_time"]
    
    # List of milestones in order (start from present day, work toward SC)
    milestones = ["SC", "SAR", "SIAR", "ASI"] # "WS"
    
    # First, handle the journey from PRESENT_DAY to SC
    if "PRESENT_DAY to SC" in samples["time_gaps"]:
        base_gap_to_sc = samples["time_gaps"]["PRESENT_DAY to SC"][sim_idx]
        remaining_gap_to_sc = max(0, base_gap_to_sc - starting_position * 365)  # Convert starting_position to days
        
        if remaining_gap_to_sc > 0:
            # Still need to complete journey to SC
            # Use speedup from PRESENT_DAY (1.02) to SC (5)
            present_day_speed = samples["speeds"].get("PRESENT_DAY", 1.02)
            sc_speed = samples["speeds"]["SC"]
            if isinstance(sc_speed, np.ndarray):
                sc_speed = sc_speed[sim_idx]
            
            calendar_days = run_phase_simulation(remaining_gap_to_sc, present_day_speed, sc_speed, progress_rate, "PRESENT_DAY to SC")
            phase_calendar_days.append(calendar_days)
            
            try:
                current_date = current_date + pd.Timedelta(days=calendar_days)
                if current_date.year > 9999:
                    current_date = datetime(9999, 12, 31)
            except (OverflowError, pd.errors.OutOfBoundsTimedelta):
                current_date = datetime(9999, 12, 31)
            
            milestone_dates.append(current_date)  # This is when SC is reached
        else:
            # Already at SC, no time needed
            milestone_dates.append(current_date)
            phase_calendar_days.append(0)
    
    # Run through each remaining milestone gap
    for i, milestone in enumerate(milestones[:-1]):
        next_milestone = milestones[i + 1]
        milestone_pair = f"{milestone} to {next_milestone}"
        gap = samples["time_gaps"][milestone_pair][sim_idx]
        
        # Get speedup values, handling SC speedup samples
        if isinstance(samples["speeds"][milestone], np.ndarray):
            start_speed = samples["speeds"][milestone][sim_idx]
        else:
            start_speed = samples["speeds"][milestone]
            
        if isinstance(samples["speeds"][next_milestone], np.ndarray):
            end_speed = samples["speeds"][next_milestone][sim_idx]
        else:
            end_speed = samples["speeds"][next_milestone]
        
        # Run simulation for this phase with exponential speedup
        calendar_days = run_phase_simulation(gap, start_speed, end_speed, progress_rate, milestone_pair)
        phase_calendar_days.append(calendar_days)
        
        try:
            current_date = current_date + pd.Timedelta(days=calendar_days)
            if current_date.year > 9999:
                current_date = datetime(9999, 12, 31)
        except (OverflowError, pd.errors.OutOfBoundsTimedelta):
            current_date = datetime(9999, 12, 31)
        
        milestone_dates.append(current_date)
        
        if current_date.year == 9999:
            for _ in range(i+1, len(milestones)-1):
                milestone_dates.append(current_date)
                phase_calendar_days.append(0)
            break
    
    return milestone_dates, phase_calendar_days

def run_single_simulation(samples: dict, sim_idx: int, progress_rate: float = 1.0) -> list[datetime]:
    """Run a single simulation and return milestone dates."""
    milestone_dates, _ = run_single_simulation_with_tracking(samples, sim_idx, progress_rate)
    return milestone_dates

def run_multi_project_simulation_with_tracking(samples: dict, sim_idx: int, project_progress_samples: dict, project_starting_positions: dict) -> tuple[dict, list[datetime], dict]:
    """Run multi-project simulation with detailed tracking.
    
    Args:
        samples: Milestone timing samples
        sim_idx: Simulation index
        project_progress_samples: Dictionary mapping project names to arrays of progress rate samples
        project_starting_positions: Dictionary mapping project names to starting positions (in years)
    
    Returns:
        Tuple of (project_results, first_milestone_dates, project_phase_durations)
    """
    project_results = {}
    project_phase_durations = {}
    
    # Run simulation for each project with tracking
    for project_name, progress_rate_samples in project_progress_samples.items():
        progress_rate = progress_rate_samples[sim_idx]  # Get rate for this simulation
        starting_position = project_starting_positions.get(project_name, 0.0)
        
        # Use the original samples (don't adjust start time)
        # Starting position will be handled inside the simulation
        milestone_dates, phase_durations = run_single_simulation_with_tracking(samples, sim_idx, progress_rate, starting_position)
        project_results[project_name] = milestone_dates
        project_phase_durations[project_name] = phase_durations
    
    # Find first achievement of each milestone across all projects
    milestones = ["SC", "SAR", "SIAR", "ASI"]  # These correspond to indices 0, 1, 2, 3 in milestone_dates
    first_milestone_dates = []
    
    for milestone_idx in range(len(milestones)):
        earliest_date = None
        for project_name in project_progress_samples:
            if milestone_idx < len(project_results[project_name]):
                project_date = project_results[project_name][milestone_idx]
                if earliest_date is None or project_date < earliest_date:
                    earliest_date = project_date
        
        if earliest_date is not None:
            first_milestone_dates.append(earliest_date)
        else:
            first_milestone_dates.append(datetime(9999, 12, 31))
    
    return project_results, first_milestone_dates, project_phase_durations

def setup_plotting_style(plotting_style: dict):
    """Set up matplotlib style according to config."""
    plt.style.use('default')  # Reset to default style
    
    # Set background color to cream (255, 250, 240)
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    plt.rcParams['figure.facecolor'] = bg_rgb
    plt.rcParams['axes.facecolor'] = bg_rgb
    plt.rcParams['savefig.facecolor'] = bg_rgb
    
    # Set font to monospace
    plt.rcParams['font.family'] = 'monospace'
    
    # Create font properties objects with sizes
    font_regular = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["title"])
    font_regular_small = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["axis_labels"])
    font_regular_legend = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["legend"])
    font_bold = fm.FontProperties(family='monospace', weight='bold', size=plotting_style["font"]["sizes"]["title"])
    font_medium = fm.FontProperties(family='monospace', weight='medium', size=plotting_style["font"]["sizes"]["title"])
    font_regular_xsmall = fm.FontProperties(family='monospace', size=plotting_style["font"]["sizes"]["small"])
    
    # Return the font properties to be used in plotting functions
    return {
        'regular': font_regular,
        'regular_small': font_regular_small,
        'regular_legend': font_regular_legend,
        'bold': font_bold,
        'medium': font_medium,
        'small': font_regular_xsmall
    }

def create_milestone_timeline_plot(all_milestone_dates: list[list[datetime]], config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create timeline plot showing milestone achievement distributions."""
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    milestone_years = [[d.year + d.timetuple().tm_yday/365 for d in sim_dates] for sim_dates in all_milestone_dates]
    milestones = ["SAR", "SIAR", "ASI"] #, "WS"]
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Initialize stats text
    stats_text = ""
    
    # Plot distribution for each milestone
    for i, milestone in enumerate(milestones):
        MAX_GRAPH_YEAR = 2032
        start_year = float(config["starting_time"].split()[-1])

        milestone_data = [years[i] for years in milestone_years]
        
        # Calculate percentiles using full data
        p10 = np.percentile(milestone_data, 10)
        p50 = np.percentile(milestone_data, 50)
        p90 = np.percentile(milestone_data, 90)
        
        # Convert decimal years to month and year format for display
        def year_to_date(year):
            year_int = int(year)
            if year_int > 9999:
                return f"{year_int}"
            month = int((year - year_int) * 12) + 1
            month_name = datetime(year_int, month, 1).strftime('%b')
            return f"{month_name} {year_int}"
        
        # Filter data to visible range for KDE
        visible_data = [x for x in milestone_data if start_year <= x <= MAX_GRAPH_YEAR]
        if not visible_data:
            print(f"Warning: No data in visible range for {milestone}")
            continue
            
        # Calculate KDE on visible data
        kde = gaussian_kde(visible_data)
        
        # Create x range for plotting
        x_range = np.linspace(start_year, MAX_GRAPH_YEAR, 200000)
        density = kde(x_range)
        
        # Normalize density to sum to 1 over visible range - fix division by zero
        density_sum = np.sum(density)
        if density_sum > 0:
            density = density / density_sum * (len(visible_data) / len(milestone_data))
        else:
            density = np.zeros_like(density)  # Fallback if density is all zeros
        
        # Plot with different colors for each milestone
        colors = ["#900000", "#004000", "#000090"]
        ax.plot(x_range, density, '-', color=colors[i], label=milestone,
                linewidth=2, alpha=0.8, zorder=2)
        ax.fill_between(x_range, density, color=colors[i], alpha=0.1)
        milestone_full = ["Superhuman\n  AI Researcher", "Superintelligent\n  AI Researcher", "Generally\n  Superintelligent"]
        # Add statistics text using full data with month and year format
        if (p90 > 2100): 
            stats = (
                f"{milestone}: {milestone_full[i]}\n"
                f"  10th: {year_to_date(p10)}\n"
                f"  50th: {year_to_date(p50)}\n"
                f"  90th: >2100"
            )
        else: 
            stats = (
                f"{milestone}: {milestone_full[i]}\n"
                f"  10th: {year_to_date(p10)}\n"
                f"  50th: {year_to_date(p50)}\n"
                f"  90th: {year_to_date(p90)}"
            )
        
        if i == 0:
            stats_text = stats
        else:
            stats_text += f"\n\n{stats}"
    
    # Add stats text with white background
    text = ax.text(0.68, 1, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            # bbox=dict(facecolor='white', alpha=0.9,
            #          edgecolor='black',
            #          linewidth=0.5),
            fontsize=plotting_style["font"]["sizes"]["small"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    title = ax.set_title("AI Takeoff Forecast, Assuming Superhuman Coder in Mar 2027",
                 fontsize=plotting_style["font"]["sizes"]["title"],
                 pad=10)
    title.set_fontproperties(fonts['regular'])
    
    xlabel = ax.set_xlabel("Year",
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    xlabel.set_fontproperties(fonts['regular_small'])
    
    ylabel = ax.set_ylabel("Probability Density",
                 fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ylabel.set_fontproperties(fonts['regular_small'])
    
    # Set axis properties
    ax.set_xlim(start_year, MAX_GRAPH_YEAR)
    ax.set_ylim(0, None)
    
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

    return fig

def create_phase_duration_plot(all_milestone_dates: list[list[datetime]], config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create box plot showing the distribution of time spent in each phase."""
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    phase_durations = []
    start_date = datetime.strptime(config["starting_time"], "%B %d %Y")
    
    for sim_dates in all_milestone_dates:
        # sim_dates should contain [SC_date, SAR_date, SIAR_date, ASI_date]
        # We want to calculate durations for: SC to SAR, SAR to SIAR, SIAR to ASI
        durations = []
        
        # Calculate durations in years between consecutive milestones
        for i in range(len(sim_dates) - 1):
            if i + 1 < len(sim_dates):  # Ensure we don't go out of bounds
                delta = sim_dates[i+1] - sim_dates[i]
                years = delta.days / 365.0
                durations.append(years)
        
        # Ensure we have exactly 3 durations (SC to SAR, SAR to SIAR, SIAR to ASI)
        # Pad with NaN if we have fewer milestones due to capping
        while len(durations) < 3:
            durations.append(float('nan'))
        
        # Take only the first 3 durations to match our phase names
        phase_durations.append(durations[:3])
    
    # Transpose to get list of durations for each phase
    phase_durations = list(map(list, zip(*phase_durations)))
    
    # Filter out NaN values for each phase
    filtered_phase_durations = []
    for phase in phase_durations:
        filtered_phase = [d for d in phase if not np.isnan(d)]
        filtered_phase_durations.append(filtered_phase)
    
    # Set up figure with space for statistics on the right
    fig = plt.figure(figsize=(12, 6), facecolor=bg_rgb)
    
    # Create two subplots - one for the boxplot, one for the stats
    gs = fig.add_gridspec(1, 2, width_ratios=[5, 1])
    ax_box = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    ax_box.set_facecolor(bg_rgb)
    ax_stats.set_facecolor(bg_rgb)
    
    # Define phases
    phase_names = ["SC to SAR", "SAR to SIAR", "SIAR to ASI"] # "ASI to WS"
    
    phase_full = ["Superhuman Coder", "Superhuman AI Researcher", "Superintelligent AI Researcher", "Artificial Superintelligence"]
    # Define shades of green for the boxes
    colors = ['#228B22', '#228B22', '#228B22']  # Dark green, Forest green, Lime green
    
    # Create box plot with log scale
    ax_box.set_yscale('log')
    
    # Create box plot with custom whiskers at 90th percentile - fix the deprecation warning
    bp = ax_box.boxplot(filtered_phase_durations, tick_labels=phase_names, patch_artist=True, 
                        whis=(10, 90))  # Set whiskers at 10th and 90th percentiles
    
    # Add color to boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Add some transparency for better visibility
    
    # Customize whiskers and medians for better visibility on log scale
    for whisker in bp['whiskers']:
        whisker.set(color='black', linewidth=1.2)
    
    for median in bp['medians']:
        median.set(color='black', linewidth=2)
    
    # Add grid appropriate for log scale
    ax_box.yaxis.grid(True, linestyle='--', alpha=0.7)
    
    # Set labels and title
    ax_box.set_ylabel("Calendar Years (log scale)", fontproperties=fonts["regular_small"])
    ax_box.set_title("Time Spent in Each Milestone Transition, assuming fixed training compute", 
                     fontproperties=fonts["regular"], loc='left')
    
    # Create statistics table in the right subplot
    ax_stats.axis('off')  # Hide axes for the stats panel
    
    # Prepare statistics text using filtered data
    stats_text = "Statistics (years):\n\n"
    for i, data in enumerate(filtered_phase_durations):
        if len(data) > 0:
            p10 = np.percentile(data, 10)
            p50 = np.percentile(data, 50)  # median
            p90 = np.percentile(data, 90)
            
            stats_text += f"{phase_full[i]} to \n {phase_full[i+1]}\n"
            stats_text += f"  10th: {p10:.2f}\n"
            stats_text += f"  50th: {p50:.2f}\n"
            if (p90 > 100): 
                stats_text += f"  90th: >100\n\n"
            else: 
                stats_text += f"  90th: {p90:.2f}\n\n"
        else:
            stats_text += f"{phase_full[i]} to \n {phase_full[i+1]}\n"
            stats_text += f"  No valid data\n\n"
    
    # Add statistics text to the right panel
    ax_stats.text(0.06, 0.95, stats_text, 
                 transform=ax_stats.transAxes,
                 verticalalignment='top',
                 fontproperties=fonts["small"],
                 bbox=dict(facecolor=bg_rgb, alpha=0.8, 
                          edgecolor='lightgray', boxstyle='round,pad=0.5'))
    
    # Set fixed y-axis ticks at 0.1, 1, 10, 100
    ax_box.set_yticks([0.01, 0.1, 1, 10, 100])
    ax_box.set_yticklabels(['0.01', '0.1', '1', '10', '100'])
    ax_box.set_ylim(0.01, 120)
    # Set the y-axis limits
    # ax_box.set_ylim(0.05, 200)  # Slightly beyond the display range for visual clarity
    
    # Add horizontal lines at each tick mark
    reference_lines = [0.01, 0.1, 1, 10, 100]
    for line_val in reference_lines:
        ax_box.axhline(y=line_val, color='gray', linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    return fig

def print_median_comparison(all_milestone_dates: list[list[datetime]], config: dict):
    """Compare median of differences vs difference of medians."""
    phase_names = ["SC to SAR", "SAR to SIAR", "SIAR to ASI"] # "ASI to WS"

    # Convert dates to years
    milestone_years = [[d.year + d.timetuple().tm_yday/365 for d in sim_dates] for sim_dates in all_milestone_dates]
    start_date = datetime.strptime(config["starting_time"], "%B %d %Y")
    start_year = start_date.year + start_date.timetuple().tm_yday/365
    
    # Calculate medians of absolute years
    medians_absolute = [np.median([years[i] for years in milestone_years if i < len(years)]) for i in range(len(phase_names))]  # Fixed number of milestones
    medians_absolute.insert(0, start_year)
    
    # Calculate differences between consecutive medians
    diff_of_medians = [medians_absolute[i+1] - medians_absolute[i] for i in range(len(medians_absolute)-1)]
    
    # Calculate phase durations for each simulation, handling potential NaN or infinite values
    phase_durations = []
    
    for sim_dates in all_milestone_dates:
        years = [d.year + d.timetuple().tm_yday/365 for d in sim_dates]
        years.insert(0, start_year)
        
        # Handle simulations with capped dates by limiting to actual number of milestones
        durations = []
        for i in range(min(len(years)-1, len(phase_names))):
            durations.append(years[i+1] - years[i])
        
        # Pad with NaN if we don't have enough values
        while len(durations) < len(phase_names):
            durations.append(float('nan'))
            
        phase_durations.append(durations)
    
    # Transpose to get list of durations for each phase
    phase_durations = list(map(list, zip(*phase_durations)))
    
    # Calculate medians, filtering out NaN or infinite values
    median_of_diffs = []
    for phase in phase_durations:
        valid_durations = [d for d in phase if np.isfinite(d)]
        if valid_durations:
            median_of_diffs.append(np.median(valid_durations))
        else:
            median_of_diffs.append(float('nan'))

def create_multi_project_timeline_plot(all_first_milestone_dates: list[list[datetime]], all_project_results: list[dict], config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create timeline plot showing milestone achievement distributions for multiple projects.
    
    Args:
        all_first_milestone_dates: List of first-to-achieve milestone dates for each simulation
        all_project_results: List of project results for each simulation
        config: Configuration dictionary
        plotting_style: Plotting style configuration
        fonts: Font configuration
    """
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    # Project comparison - show SAR distributions for each project
    projects = list(config["projects"].keys())
    project_colors = plt.cm.Set3(np.linspace(0, 1, len(projects)))
    
    MAX_GRAPH_YEAR = 2032
    start_year = float(config["starting_time"].split()[-1])
    
    # Plot SAR distributions for each project
    stats_text = ""
    for proj_idx, project_name in enumerate(projects):
        project_sar_times = []
        for sim_results in all_project_results:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]  # SAR is now index 1 (SC=0, SAR=1)
                if sar_date.year < 9999:  # Filter out capped values
                    project_sar_times.append(sar_date.year + sar_date.timetuple().tm_yday/365)
        
        if project_sar_times and len(project_sar_times) > 10:  # Need enough data for KDE
            # Calculate percentiles for stats
            p10 = np.percentile(project_sar_times, 10)
            p50 = np.percentile(project_sar_times, 50)
            p90 = np.percentile(project_sar_times, 90)
            
            # Convert decimal years to month and year format for display
            def year_to_date(year):
                year_int = int(year)
                if year_int > 9999:
                    return f"{year_int}"
                month = int((year - year_int) * 12) + 1
                month_name = datetime(year_int, month, 1).strftime('%b')
                return f"{month_name} {year_int}"
            
            # Filter data to visible range for KDE
            visible_data = [x for x in project_sar_times if start_year <= x <= MAX_GRAPH_YEAR]
            if visible_data:
                # Calculate KDE on visible data using same approach as multi-project plot
                try:
                    # Special case: if most delays are 0, add small jitter for visualization
                    if len([d for d in visible_data if d == 0]) > len(visible_data) * 0.5:
                        # Add tiny random jitter to zero values for visualization
                        jittered_data = []
                        for d in visible_data:
                            if d == 0:
                                jittered_data.append(d + np.random.normal(0, 0.01))  # Small jitter
                            else:
                                jittered_data.append(d)
                        kde = gaussian_kde(jittered_data)
                    else:
                        kde = gaussian_kde(visible_data)
                    
                    # Create x range for plotting
                    x_range = np.linspace(start_year, MAX_GRAPH_YEAR, 1000)
                    density = kde(x_range)
                    
                    # Normalize density - fix division by zero
                    density_sum = np.sum(density)
                    if density_sum > 0:
                        density = density / density_sum * (len(visible_data) / len(project_sar_times))
                    else:
                        density = np.zeros_like(density)  # Fallback if density is all zeros
                    
                    # Plot distribution
                    project_params = config["projects"][project_name]
                    if isinstance(project_params, dict) and "lower_bound" in project_params:
                        # New format - show median of range
                        median_rate = (project_params["lower_bound"] + project_params["upper_bound"]) / 2
                        rate_label = f"~{median_rate:.1f}x"
                    else:
                        # Old format - show fixed rate
                        progress_rate = project_params if isinstance(project_params, (int, float)) else 1.0
                        rate_label = f"{progress_rate:.1f}x"
                    
                    ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                           label=f"{project_name} ({rate_label})", linewidth=2, alpha=0.8)
                    ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                    
                except (np.linalg.LinAlgError, ValueError):
                    # KDE failed due to insufficient variance, use histogram instead
                    project_params = config["projects"][project_name]
                    if isinstance(project_params, dict) and "lower_bound" in project_params:
                        # New format - show median of range
                        median_rate = (project_params["lower_bound"] + project_params["upper_bound"]) / 2
                        rate_label = f"~{median_rate:.1f}x"
                    else:
                        # Old format - show fixed rate
                        progress_rate = project_params if isinstance(project_params, (int, float)) else 1.0
                        rate_label = f"{progress_rate:.1f}x"
                    
                    # Create histogram
                    bins = np.linspace(0, MAX_DELAY, 50)
                    hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Plot as step function
                    ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                           label=f"{project_name} ({rate_label})", linewidth=2, alpha=0.8)
                    ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                
                # Add to stats text
                if (p90 > 2100):
                    stats = (
                        f"{project_name} ({rate_label}):\n"
                        f"  10th: {year_to_date(p10)}\n"
                        f"  50th: {year_to_date(p50)}\n"
                        f"  90th: >2100"
                    )
                else:
                    stats = (
                        f"{project_name} ({rate_label}):\n"
                        f"  10th: {year_to_date(p10)}\n"
                        f"  50th: {year_to_date(p50)}\n"
                        f"  90th: {year_to_date(p90)}"
                    )
                
                if proj_idx == 0:
                    stats_text = stats
                else:
                    stats_text += f"\n\n{stats}"
    
    # Add stats text
    text = ax.text(0.68, 1, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=plotting_style["font"]["sizes"]["small"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    ax.set_xlim(start_year, MAX_GRAPH_YEAR)
    ax.set_ylim(0, None)
    ax.set_xlabel("Year", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Probability Density", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_title("Multi-Project Comparison: Superhuman AI Researcher Achievement", 
                 fontsize=plotting_style["font"]["sizes"]["title"], pad=10)
    
    # Add grid and styling
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper left', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=plotting_style["font"]["sizes"]["ticks"])

    return fig

def create_project_delay_plot(all_project_results: list[dict], config: dict, plotting_style: dict, fonts: dict, project_progress_samples: dict = None) -> plt.Figure:
    """Create plot showing the delay between each project's SAR achievement and the leading project's SAR achievement.
    
    Args:
        all_project_results: List of project results for each simulation
        config: Configuration dictionary
        plotting_style: Plotting style configuration
        fonts: Font configuration
        project_progress_samples: Dictionary mapping project names to arrays of progress rate samples
    """
    # Get background color
    background_color = "#FFFEF8"
    bg_rgb = tuple(int(background_color.lstrip('#')[i:i+2], 16)/255 for i in (0, 2, 4))
    
    fig = plt.figure(figsize=(12, 6), dpi=150, facecolor=bg_rgb)
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_rgb)
    
    projects = list(config["projects"].keys())
    # Include all projects in delay plot since any project could win
    projects_for_delay = projects
    project_colors = plt.cm.Set3(np.linspace(0, 1, len(projects_for_delay)))
    
    # Calculate delays for each simulation
    project_delays = {project: [] for project in projects_for_delay}
    
    for sim_idx, sim_results in enumerate(all_project_results):
        # Find the earliest SAR date across all projects in this simulation
        earliest_sar_date = None
        valid_projects = {}
        
        for project_name in projects:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]  # SAR is now index 1 (SC=0, SAR=1)
                if sar_date.year < 9999:  # Filter out capped values
                    valid_projects[project_name] = sar_date
                    if earliest_sar_date is None or sar_date < earliest_sar_date:
                        earliest_sar_date = sar_date
        
        # Calculate delays relative to earliest
        if earliest_sar_date is not None:
            for project_name in projects_for_delay:
                if project_name in valid_projects:
                    delay_days = (valid_projects[project_name] - earliest_sar_date).days
                    delay_years = delay_days / 365.0
                    project_delays[project_name].append(delay_years)
    
    # Debug: Print delay statistics
    print(f"\nDelay Statistics:")
    for project_name in projects_for_delay:
        delays = project_delays[project_name]
        if delays:
            print(f"{project_name}: min={min(delays):.2f}, max={max(delays):.2f}, mean={np.mean(delays):.2f}, count={len(delays)}")
        else:
            print(f"{project_name}: No delay data")
    
    # Debug: Check a few example simulations to see what's happening
    print(f"\nExample simulation details (first 3 simulations):")
    for sim_idx in range(min(3, len(all_project_results))):
        print(f"\nSimulation {sim_idx}:")
        sim_results = all_project_results[sim_idx]
        
        # Get progress rates for this simulation
        print("  Progress rates:")
        for project_name in projects:
            if project_name in project_progress_samples:
                rate = project_progress_samples[project_name][sim_idx]
                print(f"    {project_name}: {rate:.3f}x")
        
        # Get SAR achievement dates
        print("  SAR achievement dates:")
        sar_dates = {}
        for project_name in projects:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]
                sar_dates[project_name] = sar_date
                sar_year = sar_date.year + sar_date.timetuple().tm_yday/365
                print(f"    {project_name}: {sar_year:.2f}")
        
        # Show winner and delays
        if sar_dates:
            earliest_date = min(sar_dates.values())
            winner = [name for name, date in sar_dates.items() if date == earliest_date][0]
            print(f"  Winner: {winner}")
            print("  Delays:")
            for project_name, date in sar_dates.items():
                delay_years = (date - earliest_date).days / 365.0
                print(f"    {project_name}: {delay_years:.3f} years")
    
    # Plot delay distributions for each project
    stats_text = ""
    has_plotted_data = False
    
    for proj_idx, project_name in enumerate(projects_for_delay):
        delays = project_delays[project_name]
        
        if delays and len(delays) > 1:  # Need enough data for KDE
            # Calculate percentiles for stats
            p10 = np.percentile(delays, 10)
            p50 = np.percentile(delays, 50)
            p90 = np.percentile(delays, 90)
            
            # Check if this project has meaningful delay variation
            delay_variation = p90 - p10
            
            # Get project parameters for labeling
            project_params = config["projects"][project_name]
            if isinstance(project_params, dict) and "lower_bound" in project_params:
                # New format - show median of range
                median_rate = (project_params["lower_bound"] + project_params["upper_bound"]) / 2
                rate_label = f"~{median_rate:.1f}x"
            else:
                # Old format - show fixed rate
                progress_rate = project_params if isinstance(project_params, (int, float)) else 1.0
                rate_label = f"{progress_rate:.1f}x"
            
            # Special handling for projects that always (or almost always) win (very small delays)
            if delay_variation < 0.1:  # Less than 0.1 years variation
                # Add a vertical line at the median delay position instead of a distribution
                ax.axvline(x=p50, color=project_colors[proj_idx], linestyle='--', linewidth=2, 
                          alpha=0.8, label=f"{project_name} ({rate_label}) - Always wins")
                
                # Add to stats text
                if p50 < 0.01:
                    stats = f"{project_name} ({rate_label}):\n  Always wins (0.00 yrs)"
                else:
                    stats = f"{project_name} ({rate_label}):\n  Consistent: {p50:.2f} yrs"
            else:
                # Plot normal distribution for projects with meaningful delay variation
                # Determine appropriate x-axis range based on the data
                max_delay_in_data = max(delays)
                # For small delays, ensure minimum visible range of 2 years
                MAX_DELAY = max(2.0, min(max(5, max_delay_in_data * 1.2), 20))  # Ensure at least 2 years range
                
                # Filter data to reasonable range for visualization (include all data since delays are small)
                visible_data = delays  # Don't filter small delays out
                
                if visible_data and len(visible_data) > 1:
                    # Calculate KDE on visible data using same approach as multi-project plot
                    try:
                        # For projects with some variation, use standard KDE
                        kde = gaussian_kde(visible_data)
                        
                        # Create x range for plotting (same resolution as multi-project plot)
                        x_range = np.linspace(0, MAX_DELAY, 1000)
                        density = kde(x_range)
                        
                        # Normalize density (same approach as multi-project plot) - fix division by zero
                        density_sum = np.sum(density)
                        if density_sum > 0:
                            density = density / density_sum * (len(visible_data) / len(delays))
                        else:
                            density = np.zeros_like(density)  # Fallback if density is all zeros
                        
                        # Plot distribution
                        ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                               label=f"{project_name} ({rate_label})", linewidth=2, alpha=0.8)
                        ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                        has_plotted_data = True
                        
                    except (np.linalg.LinAlgError, ValueError):
                        # KDE failed, use histogram instead
                        # Create histogram
                        bins = np.linspace(min(visible_data), max(visible_data), 20)
                        hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                        
                        # Plot as step function
                        ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                               label=f"{project_name} ({rate_label})", linewidth=2, alpha=0.8)
                        ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                    
                    # Add to stats text
                    if p90 > 20:  # Adjust threshold based on reasonable delays
                        stats = (
                            f"{project_name} ({rate_label}):\n"
                            f"  10th: {p10:.2f} yrs\n"
                            f"  50th: {p50:.2f} yrs\n"
                            f"  90th: >20 yrs"
                        )
                    else:
                        stats = (
                            f"{project_name} ({rate_label}):\n"
                            f"  10th: {p10:.2f} yrs\n"
                            f"  50th: {p50:.2f} yrs\n"
                            f"  90th: {p90:.2f} yrs"
                        )
                
            # Add to stats text
            if proj_idx == 0:
                stats_text = stats
            else:
                stats_text += f"\n\n{stats}"
    
    # Determine final x-axis limit based on actual data, excluding always-zero delays
    non_zero_delays = [delay for delays in project_delays.values() for delay in delays if delay > 0.01]
    if non_zero_delays:
        data_max = np.percentile(non_zero_delays, 95)  # Use 95th percentile to avoid outliers
        # For small delays, ensure minimum visible range of 2 years, but focus on actual variation
        final_xlim = max(2.0, min(data_max * 1.3, 15))  # Reasonable range
    else:
        final_xlim = 5
    
    # Add stats text
    text = ax.text(0.68, 1, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='left',
            fontsize=plotting_style["font"]["sizes"]["small"])
    text.set_fontproperties(fonts['regular_legend'])
    
    # Configure plot styling
    ax.set_xlim(0, final_xlim)
    ax.set_ylim(0, None)
    ax.set_xlabel("Delay Behind Leading Project (Years)", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_ylabel("Probability Density", fontsize=plotting_style["font"]["sizes"]["axis_labels"])
    ax.set_title("Project SAR Achievement Delays Relative to Leading Project", 
                 fontsize=plotting_style["font"]["sizes"]["title"], pad=10)
    
    # Add grid and styling
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add legend
    legend = ax.legend(loc='upper right', fontsize=plotting_style["font"]["sizes"]["legend"])
    for text in legend.get_texts():
        text.set_fontproperties(fonts['regular_legend'])
    
    # Configure ticks
    ax.tick_params(axis="both", labelsize=plotting_style["font"]["sizes"]["ticks"])

    return fig

def run_takeoff_simulation(config_path: str = "takeoff_params.yaml") -> tuple[plt.Figure, dict]:
    """Run takeoff simulation and create visualizations."""
    print("Loading configuration...")
    config = load_config(config_path)
    plotting_style = config["plotting_style"]
    
    # Set up fonts
    fonts = setup_plotting_style(plotting_style)
    
    # Generate samples
    print("\nGenerating samples...")
    samples = get_milestone_samples(config, config["simulation"]["n_sims"])
    
    # Run simulations
    print("\nRunning simulations...")
    all_milestone_dates = []
    all_phase_durations = []
    for i in tqdm(range(config["simulation"]["n_sims"]), desc="Simulations"):
        milestone_dates, phase_durations = run_single_simulation_with_tracking(samples, i)
        all_milestone_dates.append(milestone_dates)
        all_phase_durations.append(phase_durations)
    
    # Validate against explicit recalculation
    validate_phase_durations(all_milestone_dates, all_phase_durations, samples["start_time"])
    
    # Print median comparison
    print_median_comparison(all_milestone_dates, config)
    
    # Create plots
    print("\nGenerating plots...")
    fig_timeline = create_milestone_timeline_plot(all_milestone_dates, config, plotting_style, fonts)
    fig_phases = create_phase_duration_plot(all_milestone_dates, config, plotting_style, fonts)
    
    # Create output directory if it doesn't exist
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving plots...")
    fig_timeline.savefig(output_dir / "takeoff_timeline.png", dpi=300, bbox_inches="tight")
    fig_phases.savefig(output_dir / "phase_durations.png", dpi=300, bbox_inches="tight")
    
    # Close figures to free memory
    plt.close("all")
    
    return fig_timeline, {"milestone_dates": all_milestone_dates}

def validate_phase_durations(all_milestone_dates, all_phase_durations, start_date):
    """Validate phase durations from simulation match direct calculation.
    
    This compares the phase durations stored during simulation with
    those calculated by just looking at the difference between milestone dates.
    """
    print("\nValidating phase durations:")
    
    # Calculate durations directly from milestone dates
    calculated_durations = []
    for milestone_dates in all_milestone_dates:
        # Include start date
        dates = [start_date] + milestone_dates
        durations = []
        
        # Calculate and handle potential overflow
        for i in range(len(dates)-1):
            try:
                delta = dates[i+1] - dates[i]
                durations.append(delta.days)
            except (OverflowError, pd.errors.OutOfBoundsTimedelta):
                durations.append(float('inf'))  # Represent as infinity
        
        calculated_durations.append(durations)
    
    # Compare with stored durations
    first_mismatch = None
    
    for sim_idx in range(len(all_milestone_dates)):
        for phase_idx in range(len(all_phase_durations[sim_idx])):
            stored = all_phase_durations[sim_idx][phase_idx]
            calculated = calculated_durations[sim_idx][phase_idx]
            
            # For capped values, the calculated might be infinity
            if calculated == float('inf'):
                continue
                
            # Due to floating point, there might be tiny differences
            if abs(stored - calculated) > 0.1:  # Allow small difference
                if first_mismatch is None:
                    first_mismatch = (sim_idx, phase_idx, stored, calculated)

def run_multi_project_takeoff_simulation(config_path: str = "takeoff_params.yaml") -> tuple[plt.Figure, dict]:
    """Run multi-project takeoff simulation and create visualizations."""
    print("Loading configuration...")
    config = load_config(config_path)
    plotting_style = config["plotting_style"]
    
    # Check if projects are defined in config
    if "projects" not in config:
        print("No projects defined in config. Running single-project simulation instead.")
        return run_takeoff_simulation(config_path)
    
    # Generate project progress rate samples
    print("\nGenerating project progress rate samples...")
    project_progress_samples = get_project_progress_samples(config, config["simulation"]["n_sims"])
    
    # Get project starting positions
    project_starting_positions = get_project_starting_positions(config)
    
    # Print statistics for progress rates and starting positions
    print("Project configurations:")
    for project_name, samples in project_progress_samples.items():
        p10 = np.percentile(samples, 10)
        p50 = np.percentile(samples, 50)
        p90 = np.percentile(samples, 90)
        starting_pos = project_starting_positions.get(project_name, 0.0)
        print(f"  {project_name}: progress rates 10th={p10:.2f}x, 50th={p50:.2f}x, 90th={p90:.2f}x, starting position={starting_pos:.1f} years")
    
    # Set up fonts
    fonts = setup_plotting_style(plotting_style)
    
    # Generate samples (shared across all projects)
    print("\nGenerating milestone samples...")
    samples = get_milestone_samples(config, config["simulation"]["n_sims"])
    
    # Run multi-project simulations
    print("\nRunning multi-project simulations...")
    all_first_milestone_dates = []
    all_project_results = []
    all_project_phase_durations = []
    
    for i in tqdm(range(config["simulation"]["n_sims"]), desc="Simulations"):
        project_results, first_milestone_dates, project_phase_durations = run_multi_project_simulation_with_tracking(samples, i, project_progress_samples, project_starting_positions)
        all_first_milestone_dates.append(first_milestone_dates)
        all_project_results.append(project_results)
        all_project_phase_durations.append(project_phase_durations)
    
    # Print summary statistics
    print("\nFirst-to-achieve milestone statistics:")
    milestones = ["SC", "SAR", "SIAR", "ASI"]
    for i, milestone in enumerate(milestones):
        milestone_years = [dates[i].year + dates[i].timetuple().tm_yday/365 
                          for dates in all_first_milestone_dates 
                          if i < len(dates) and dates[i].year < 9999]
        
        if milestone_years:
            p10 = np.percentile(milestone_years, 10)
            p50 = np.percentile(milestone_years, 50)
            p90 = np.percentile(milestone_years, 90)
            print(f"{milestone}: 10th={p10:.1f}, 50th={p50:.1f}, 90th={p90:.1f}")
        else:
            print(f"{milestone}: No valid data")
    
    # Analysis: Check which project wins SAR most often
    print("\nSAR Winner Analysis:")
    sar_winners = {}
    projects = list(config["projects"].keys())
    
    for sim_idx, sim_results in enumerate(all_project_results):
        earliest_sar_date = None
        winner = None
        
        for project_name in projects:
            if project_name in sim_results and len(sim_results[project_name]) > 1:
                sar_date = sim_results[project_name][1]  # SAR is now index 1 (SC=0, SAR=1)
                if sar_date.year < 9999:  # Filter out capped values
                    if earliest_sar_date is None or sar_date < earliest_sar_date:
                        earliest_sar_date = sar_date
                        winner = project_name
        
        if winner:
            sar_winners[winner] = sar_winners.get(winner, 0) + 1
    
    total_wins = sum(sar_winners.values())
    print(f"Total valid simulations: {total_wins}")
    for project, wins in sorted(sar_winners.items(), key=lambda x: x[1], reverse=True):
        percentage = (wins / total_wins) * 100 if total_wins > 0 else 0
        print(f"  {project}: {wins} wins ({percentage:.1f}%)")
    
    leading_lab_always_wins = sar_winners.get("Leading Lab", 0) == total_wins
    print(f"\nDoes Leading Lab always win SAR? {leading_lab_always_wins}")
    if not leading_lab_always_wins:
        print("-> Other projects sometimes beat Leading Lab!")
    else:
        print("-> Leading Lab wins every time")
    
    # Create plots
    print("\nGenerating plots...")
    fig_multi_timeline = create_multi_project_timeline_plot(all_first_milestone_dates, all_project_results, config, plotting_style, fonts)
    fig_project_delays = create_project_delay_plot(all_project_results, config, plotting_style, fonts, project_progress_samples)
    
    # Also create single-project plots for the fastest project (for comparison)
    fastest_project = min(project_progress_samples.keys(), key=lambda x: 1/np.mean(project_progress_samples[x]))  # Highest mean progress rate
    fastest_milestone_dates = [results[fastest_project] for results in all_project_results]
    fig_fastest_timeline = create_milestone_timeline_plot(fastest_milestone_dates, config, plotting_style, fonts)
    fig_fastest_phases = create_phase_duration_plot(fastest_milestone_dates, config, plotting_style, fonts)
    
    # Create output directory if it doesn't exist
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nSaving plots...")
    fig_multi_timeline.savefig(output_dir / "multi_project_takeoff_timeline.png", dpi=300, bbox_inches="tight")
    fig_project_delays.savefig(output_dir / "project_sar_delays.png", dpi=300, bbox_inches="tight")
    fig_fastest_timeline.savefig(output_dir / f"fastest_project_{fastest_project.replace(' ', '_')}_timeline.png", dpi=300, bbox_inches="tight")
    fig_fastest_phases.savefig(output_dir / f"fastest_project_{fastest_project.replace(' ', '_')}_phases.png", dpi=300, bbox_inches="tight")
    
    # Close figures to free memory
    plt.close("all")
    
    return fig_multi_timeline, {
        "first_milestone_dates": all_first_milestone_dates,
        "project_results": all_project_results,
        "project_phase_durations": all_project_phase_durations
    }

if __name__ == "__main__":
    run_multi_project_takeoff_simulation()
    print(f"\nSimulation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}") 