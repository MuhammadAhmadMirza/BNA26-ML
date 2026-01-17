"""
Server Performance Dataset Generator

Generates a realistic dataset simulating server performance metrics with
power_consumption as the target variable. The relationships between features
and target are designed to be:
- Realistic and natural (based on real-world server behavior)
- Complex and non-linear (polynomial, interaction, and threshold effects)
- Balanced for analysis visualizations
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
N_SAMPLES = 2000
OUTPUT_PATH = "../data/dataset.csv"


def generate_base_features(n: int) -> dict:
    """Generate base server performance features with realistic distributions."""
    
    # CPU utilization: bimodal distribution (servers are often idle or busy)
    cpu_low = np.random.beta(2, 5, n // 3) * 40  # Idle servers: 0-40%
    cpu_mid = np.random.beta(5, 5, n // 3) * 40 + 30  # Moderate: 30-70%
    cpu_high = np.random.beta(5, 2, n - 2 * (n // 3)) * 40 + 60  # Busy: 60-100%
    cpu_utilization = np.clip(np.concatenate([cpu_low, cpu_mid, cpu_high]), 5, 100)
    np.random.shuffle(cpu_utilization)
    
    # Memory usage: correlated with CPU but with independent variation
    memory_base = 0.5 * cpu_utilization + np.random.normal(20, 15, n)
    memory_usage = np.clip(memory_base, 5, 95)
    
    # Disk I/O: partially correlated with CPU and memory, with spikes
    disk_base = 0.8 * cpu_utilization + 0.4 * memory_usage + np.random.exponential(15, n)
    disk_io = np.clip(disk_base, 20, 250)
    
    # Network latency: mostly independent, with occasional spikes (log-normal)
    network_latency = np.clip(np.random.lognormal(2.8, 0.6, n), 5, 100)
    
    # Process count: correlated with CPU utilization
    process_base = 30 + 2.2 * cpu_utilization + np.random.normal(0, 25, n)
    process_count = np.clip(process_base, 20, 350).astype(int)
    
    # Thread count: correlated with process count (threads per process varies)
    threads_per_process = np.random.uniform(3, 12, n)
    thread_count = np.clip(process_count * threads_per_process + np.random.normal(0, 100, n), 
                           100, 2500).astype(int)
    
    # Context switches: strongly correlated with threads and CPU
    context_base = 3 * thread_count + 40 * cpu_utilization + np.random.normal(0, 500, n)
    context_switches = np.clip(context_base, 1000, 20000).astype(int)
    
    # Cache miss rate: inversely related to memory efficiency, higher under load
    cache_base = 5 + 0.15 * cpu_utilization + 0.08 * memory_usage + np.random.exponential(3, n)
    cache_miss_rate = np.clip(cache_base, 2, 35)
    
    # Temperature: strongly correlated with CPU, non-linear at high loads
    temp_base = 30 + 0.35 * cpu_utilization + 0.05 * (cpu_utilization ** 1.3) / 10
    temp_noise = np.random.normal(0, 3, n)
    temperature = np.clip(temp_base + temp_noise, 30, 90)
    
    # Uptime: mostly independent (log-uniform distribution for realistic server uptimes)
    uptime = np.exp(np.random.uniform(np.log(1), np.log(15000), n))
    
    return {
        'cpu_utilization': cpu_utilization,
        'memory_usage': memory_usage,
        'disk_io': disk_io,
        'network_latency': network_latency,
        'process_count': process_count,
        'thread_count': thread_count,
        'context_switches': context_switches,
        'cache_miss_rate': cache_miss_rate,
        'temperature': temperature,
        'uptime': uptime
    }


def calculate_power_consumption(features: dict) -> np.ndarray:
    """
    Calculate power consumption based on realistic server power models.
    
    Power consumption in servers is influenced by:
    - CPU (dominant factor, non-linear at high loads)
    - Memory (moderate contribution)
    - Disk I/O (smaller but measurable)
    - Temperature effects (efficiency drops at high temps)
    - Context switching overhead
    - Baseline idle power
    - Hidden/unobserved factors (significant noise component)
    """
    n = len(features['cpu_utilization'])
    
    # Base idle power (servers consume significant power even when idle)
    base_power = 150
    
    # CPU contribution: non-linear (polynomial component for high utilization)
    cpu = features['cpu_utilization']
    cpu_power = 0.8 * cpu + 0.025 * (cpu ** 1.7)  # More non-linear
    
    # Memory contribution: non-linear with threshold
    mem = features['memory_usage']
    mem_power = 0.2 * mem + 0.008 * (mem ** 1.6) + np.where(mem > 70, 0.5 * (mem - 70), 0)
    
    # Disk I/O contribution: non-linear
    disk = features['disk_io']
    disk_power = 0.04 * disk + 0.0004 * (disk ** 1.5)
    
    # Temperature penalty: non-linear efficiency drops at high temperatures
    temp = features['temperature']
    temp_penalty = np.where(temp > 50, 0.4 * (temp - 50) ** 1.25, 0)
    
    # Context switching overhead: non-linear at high counts
    ctx = features['context_switches']
    ctx_power = 0.001 * ctx + 0.00000005 * (ctx ** 1.8)
    
    # Cache miss penalty: non-linear effect
    cache = features['cache_miss_rate']
    cache_power = 0.3 * cache + 0.02 * (cache ** 1.4)
    
    # Thread overhead: non-linear
    threads = features['thread_count']
    thread_power = 0.002 * threads + 0.000001 * (threads ** 1.5)
    
    # Process overhead: non-linear
    procs = features['process_count']
    proc_power = 0.01 * procs + 0.00005 * (procs ** 1.6)
    
    # Network latency effect: threshold-based non-linearity
    net = features['network_latency']
    net_power = 0.05 * net + np.where(net > 40, 0.3 * (net - 40) ** 1.2, 0)
    
    # Uptime effect: logarithmic (diminishing effect over time)
    uptime = features['uptime']
    uptime_effect = 5 * np.log1p(uptime / 1000)
    
    # Complex interaction effects
    # High CPU + high memory = synergistic power draw
    cpu_mem_interaction = 0.015 * (cpu / 50) * (mem / 50) * np.sqrt(cpu * mem / 100)
    
    # High disk I/O under high CPU load
    cpu_disk_interaction = 0.0008 * (cpu / 50) * (disk / 100) * np.sqrt(cpu)
    
    # Temperature-CPU interaction: hot CPUs under load consume more
    temp_cpu_interaction = np.where(temp > 55, 0.004 * cpu * np.sqrt(temp - 55), 0)
    
    # Complex non-linear hidden factor (simulates unobserved behavior)
    hidden_factor = (20 * np.sin(cpu / 25) * np.cos(mem / 35) + 
                    15 * np.sin(disk / 40) * np.sin(temp / 30) +
                    10 * np.cos(ctx / 5000) * np.sin(threads / 500))
    
    # Calculate total power
    power = (base_power + 
             cpu_power + 
             mem_power + 
             disk_power + 
             temp_penalty + 
             ctx_power + 
             cache_power + 
             thread_power + 
             proc_power +
             net_power +
             uptime_effect +
             cpu_mem_interaction + 
             cpu_disk_interaction +
             temp_cpu_interaction +
             hidden_factor)
    
    # Add significant realistic noise (measurement error and unmodeled factors)
    # Larger noise reduces linearity
    noise = np.random.normal(0, 25, n)
    power = power + noise
    
    # Add heteroscedastic noise (more variance at higher power levels)
    heteroscedastic_noise = np.random.normal(0, 1, n) * (power / 40)
    power = power + heteroscedastic_noise
    
    # Random spikes (unmodeled server events) - more frequent
    spike_mask = np.random.random(n) < 0.05
    power[spike_mask] += np.random.uniform(-35, 45, spike_mask.sum())
    
    # Clip to realistic server power range (150W idle to 450W max)
    power = np.clip(power, 150, 450)
    
    return power


def add_outliers_and_anomalies(df: pd.DataFrame, fraction: float = 0.03) -> pd.DataFrame:
    """Add realistic outliers and anomalies to the dataset."""
    n_outliers = int(len(df) * fraction)
    outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
    
    for idx in outlier_indices:
        # Random anomaly type
        anomaly_type = np.random.choice(['cpu_spike', 'memory_leak', 'io_storm', 'thermal_throttle'])
        
        if anomaly_type == 'cpu_spike':
            df.loc[idx, 'cpu_utilization'] = np.random.uniform(95, 100)
            df.loc[idx, 'temperature'] = np.clip(df.loc[idx, 'temperature'] + 10, 30, 90)
        elif anomaly_type == 'memory_leak':
            df.loc[idx, 'memory_usage'] = np.random.uniform(90, 98)
            df.loc[idx, 'cache_miss_rate'] = np.clip(df.loc[idx, 'cache_miss_rate'] * 1.5, 2, 35)
        elif anomaly_type == 'io_storm':
            df.loc[idx, 'disk_io'] = np.random.uniform(200, 250)
            df.loc[idx, 'context_switches'] = int(min(df.loc[idx, 'context_switches'] * 1.3, 20000))
        elif anomaly_type == 'thermal_throttle':
            df.loc[idx, 'temperature'] = np.random.uniform(80, 90)
            df.loc[idx, 'cpu_utilization'] = np.clip(df.loc[idx, 'cpu_utilization'] * 0.8, 5, 100)
    
    return df


def round_features(df: pd.DataFrame) -> pd.DataFrame:
    """Round features to realistic precision levels."""
    df['cpu_utilization'] = df['cpu_utilization'].round(2)
    df['memory_usage'] = df['memory_usage'].round(2)
    df['disk_io'] = df['disk_io'].round(2)
    df['network_latency'] = df['network_latency'].round(2)
    df['process_count'] = df['process_count'].astype(int)
    df['thread_count'] = df['thread_count'].astype(int)
    df['context_switches'] = df['context_switches'].astype(int)
    df['cache_miss_rate'] = df['cache_miss_rate'].round(2)
    df['temperature'] = df['temperature'].round(2)
    df['uptime'] = df['uptime'].round(2)
    df['power_consumption'] = df['power_consumption'].round(2)
    return df


def generate_dataset(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Generate the complete server performance dataset."""
    
    # Generate base features
    features = generate_base_features(n_samples)
    
    # Calculate target variable
    power_consumption = calculate_power_consumption(features)
    
    # Create DataFrame
    df = pd.DataFrame({
        'cpu_utilization': features['cpu_utilization'],
        'memory_usage': features['memory_usage'],
        'disk_io': features['disk_io'],
        'network_latency': features['network_latency'],
        'process_count': features['process_count'],
        'thread_count': features['thread_count'],
        'context_switches': features['context_switches'],
        'cache_miss_rate': features['cache_miss_rate'],
        'temperature': features['temperature'],
        'uptime': features['uptime'],
        'power_consumption': power_consumption
    })
    
    # Add outliers and anomalies
    df = add_outliers_and_anomalies(df)
    
    # Recalculate power for anomalous rows to maintain relationship
    anomaly_mask = (df['cpu_utilization'] > 95) | (df['memory_usage'] > 90) | (df['disk_io'] > 200)
    if anomaly_mask.any():
        anomaly_features = {col: df.loc[anomaly_mask, col].values 
                          for col in features.keys()}
        df.loc[anomaly_mask, 'power_consumption'] = calculate_power_consumption(anomaly_features)
    
    # Round to realistic precision
    df = round_features(df)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def main():
    """Main function to generate and save the dataset."""
    print("Generating server performance dataset...")
    
    # Generate dataset
    df = generate_dataset(N_SAMPLES)
    
    # Create output directory if needed
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save to CSV
    df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nDataset Statistics:")
    print(df.describe().round(2))
    
    # Print correlation with target
    print(f"\nCorrelations with power_consumption:")
    correlations = df.corr()['power_consumption'].drop('power_consumption').sort_values(ascending=False)
    for feature, corr in correlations.items():
        print(f"  {feature}: {corr:.3f}")


if __name__ == "__main__":
    main()
