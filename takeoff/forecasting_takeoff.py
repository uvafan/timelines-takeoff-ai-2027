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

def run_single_simulation_with_tracking(samples: dict, sim_idx: int, progress_rate: float = 1.0) -> tuple[list[datetime], list[float]]:
    """Run a single simulation and track both milestone dates and phase durations."""
    milestone_dates = []
    phase_calendar_days = []
    current_date = samples["start_time"]
    
    # List of milestones in order
    milestones = ["SC", "SAR", "SIAR", "ASI"] # "WS"
    
    # Run through each milestone gap
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

def run_multi_project_simulation_with_tracking(samples: dict, sim_idx: int, projects: dict) -> tuple[dict, list[datetime], dict]:
    """Run multi-project simulation with detailed tracking.
    
    Returns:
        Tuple of (project_results, first_milestone_dates, project_phase_durations)
    """
    project_results = {}
    project_phase_durations = {}
    
    # Run simulation for each project with tracking
    for project_name, progress_rate in projects.items():
        milestone_dates, phase_durations = run_single_simulation_with_tracking(samples, sim_idx, progress_rate)
        project_results[project_name] = milestone_dates
        project_phase_durations[project_name] = phase_durations
    
    # Find first achievement of each milestone across all projects
    milestones = ["SAR", "SIAR", "ASI"]
    first_milestone_dates = []
    
    for milestone_idx in range(len(milestones)):
        earliest_date = None
        for project_name in projects:
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
        
        # Normalize density to sum to 1 over visible range
        density = density / np.sum(density) * (len(visible_data) / len(milestone_data))
        
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
        # Include start date
        dates = [start_date] + sim_dates
        durations = []
        
        # Calculate durations in years
        for i in range(len(dates) - 1):
            delta = dates[i+1] - dates[i]
            years = delta.days / 365.0
            durations.append(years)
        
        phase_durations.append(durations)
    
    # Transpose to get list of durations for each phase
    phase_durations = list(map(list, zip(*phase_durations)))
    
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
    
    # Create box plot with custom whiskers at 90th percentile
    bp = ax_box.boxplot(phase_durations, labels=phase_names, patch_artist=True, 
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
    
    # Prepare statistics text using full data
    stats_text = "Statistics (years):\n\n"
    for i, data in enumerate(zip(phase_durations)):
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
            if project_name in sim_results and len(sim_results[project_name]) > 0:
                sar_date = sim_results[project_name][0]  # SAR is first milestone
                sar_year = sar_date.year + sar_date.timetuple().tm_yday/365
                if sar_year < 2100:  # Filter out capped values
                    project_sar_times.append(sar_year)
        
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
                # Calculate KDE on visible data
                try:
                    kde = gaussian_kde(visible_data)
                    
                    # Create x range for plotting
                    x_range = np.linspace(start_year, MAX_GRAPH_YEAR, 1000)
                    density = kde(x_range)
                    
                    # Normalize density
                    density = density / np.sum(density) * (len(visible_data) / len(project_sar_times))
                    
                    # Plot distribution
                    progress_rate = config["projects"][project_name]
                    label = f"{project_name} ({progress_rate:.1f}x)"
                    ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                           label=label, linewidth=2, alpha=0.8)
                    ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                    
                except (np.linalg.LinAlgError, ValueError):
                    # KDE failed due to insufficient variance, use histogram instead
                    progress_rate = config["projects"][project_name]
                    label = f"{project_name} ({progress_rate:.1f}x)"
                    
                    # Create histogram
                    bins = np.linspace(0, MAX_DELAY, 50)
                    hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Plot as step function
                    ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                           label=label, linewidth=2, alpha=0.8)
                    ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                
                # Add to stats text
                if (p90 > 2100):
                    stats = (
                        f"{project_name} ({progress_rate:.1f}x):\n"
                        f"  10th: {year_to_date(p10)}\n"
                        f"  50th: {year_to_date(p50)}\n"
                        f"  90th: >2100"
                    )
                else:
                    stats = (
                        f"{project_name} ({progress_rate:.1f}x):\n"
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

def create_project_delay_plot(all_project_results: list[dict], config: dict, plotting_style: dict, fonts: dict) -> plt.Figure:
    """Create plot showing the delay between each project's SAR achievement and the leading project's SAR achievement.
    
    Args:
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
    
    projects = list(config["projects"].keys())
    # Exclude Leading Lab from delay plot since it always has 0 delay
    projects_for_delay = [p for p in projects if p != "Leading Lab"]
    project_colors = plt.cm.Set3(np.linspace(0, 1, len(projects_for_delay)))
    
    # Calculate delays for each simulation
    project_delays = {project: [] for project in projects_for_delay}
    
    for sim_idx, sim_results in enumerate(all_project_results):
        # Find the earliest SAR date across all projects in this simulation
        earliest_sar_date = None
        valid_projects = {}
        
        for project_name in projects:
            if project_name in sim_results and len(sim_results[project_name]) > 0:
                sar_date = sim_results[project_name][0]  # SAR is first milestone
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
    
    # Plot delay distributions for each project
    stats_text = ""
    for proj_idx, project_name in enumerate(projects_for_delay):
        delays = project_delays[project_name]
        
        if delays and len(delays) > 10:  # Need enough data for KDE
            # Calculate percentiles for stats
            p10 = np.percentile(delays, 10)
            p50 = np.percentile(delays, 50)
            p90 = np.percentile(delays, 90)
            
            # Determine appropriate x-axis range based on the data
            max_delay_in_data = max(delays)
            MAX_DELAY = min(max(5, max_delay_in_data * 1.1), 20)  # Adaptive range, cap at 20 years
            
            # Filter data to reasonable range for visualization
            visible_data = [x for x in delays if 0 <= x <= MAX_DELAY]
            
            if visible_data:
                # Calculate KDE on visible data using same approach as multi-project plot
                try:
                    kde = gaussian_kde(visible_data)
                    
                    # Create x range for plotting (same resolution as multi-project plot)
                    x_range = np.linspace(0, MAX_DELAY, 1000)
                    density = kde(x_range)
                    
                    # Normalize density (same approach as multi-project plot)
                    density = density / np.sum(density) * (len(visible_data) / len(delays))
                    
                    # Plot distribution
                    progress_rate = config["projects"][project_name]
                    label = f"{project_name} ({progress_rate:.1f}x)"
                    ax.plot(x_range, density, '-', color=project_colors[proj_idx], 
                           label=label, linewidth=2, alpha=0.8)
                    ax.fill_between(x_range, density, color=project_colors[proj_idx], alpha=0.2)
                    
                except (np.linalg.LinAlgError, ValueError):
                    # KDE failed, use histogram instead
                    progress_rate = config["projects"][project_name]
                    label = f"{project_name} ({progress_rate:.1f}x)"
                    
                    # Create histogram
                    bins = np.linspace(0, MAX_DELAY, 50)
                    hist, bin_edges = np.histogram(visible_data, bins=bins, density=True)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Plot as step function
                    ax.step(bin_centers, hist, where='mid', color=project_colors[proj_idx], 
                           label=label, linewidth=2, alpha=0.8)
                    ax.fill_between(bin_centers, hist, step='mid', color=project_colors[proj_idx], alpha=0.2)
                
                # Add to stats text
                if p90 > MAX_DELAY:
                    stats = (
                        f"{project_name} ({progress_rate:.1f}x):\n"
                        f"  10th: {p10:.2f} yrs\n"
                        f"  50th: {p50:.2f} yrs\n"
                        f"  90th: >{MAX_DELAY:.0f} yrs"
                    )
                else:
                    stats = (
                        f"{project_name} ({progress_rate:.1f}x):\n"
                        f"  10th: {p10:.2f} yrs\n"
                        f"  50th: {p50:.2f} yrs\n"
                        f"  90th: {p90:.2f} yrs"
                    )
                
                if proj_idx == 0:
                    stats_text = stats
                else:
                    stats_text += f"\n\n{stats}"
    
    # Determine final x-axis limit based on actual data
    all_delays = [delay for delays in project_delays.values() for delay in delays]
    if all_delays:
        data_max = np.percentile(all_delays, 95)  # Use 95th percentile to avoid outliers
        final_xlim = min(max(3, data_max * 1.2), 15)  # Reasonable range
    else:
        final_xlim = 10
    
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
    all_match = True
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
                all_match = False
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
    
    projects = config["projects"]
    print(f"Running simulation with {len(projects)} projects:")
    for name, rate in projects.items():
        print(f"  {name}: {rate:.1f}x progress rate")
    
    # Set up fonts
    fonts = setup_plotting_style(plotting_style)
    
    # Generate samples (shared across all projects)
    print("\nGenerating samples...")
    samples = get_milestone_samples(config, config["simulation"]["n_sims"])
    
    # Run multi-project simulations
    print("\nRunning multi-project simulations...")
    all_first_milestone_dates = []
    all_project_results = []
    all_project_phase_durations = []
    
    for i in tqdm(range(config["simulation"]["n_sims"]), desc="Simulations"):
        project_results, first_milestone_dates, project_phase_durations = run_multi_project_simulation_with_tracking(samples, i, projects)
        all_first_milestone_dates.append(first_milestone_dates)
        all_project_results.append(project_results)
        all_project_phase_durations.append(project_phase_durations)
    
    # Print summary statistics
    print("\nFirst-to-achieve milestone statistics:")
    milestones = ["SAR", "SIAR", "ASI"]
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
    
    # Create plots
    print("\nGenerating plots...")
    fig_multi_timeline = create_multi_project_timeline_plot(all_first_milestone_dates, all_project_results, config, plotting_style, fonts)
    fig_project_delays = create_project_delay_plot(all_project_results, config, plotting_style, fonts)
    
    # Also create single-project plots for the fastest project (for comparison)
    fastest_project = min(projects.keys(), key=lambda x: 1/projects[x])  # Highest progress rate
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