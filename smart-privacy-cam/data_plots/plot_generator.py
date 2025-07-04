"""
Plot Generator Module
--------------------
Generates comprehensive data visualizations and analytics plots
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available. Using default matplotlib styling.")
from datetime import datetime, timedelta
import os

class PlotGenerator:
    """
    PlotGenerator.__init__
    ---------------------
    1. Initialize plotting configuration
    2. Set up matplotlib style and color schemes
    """
    def __init__(self):
        # Set up matplotlib style
        plt.style.use('dark_background')
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
        
        # Ensure data_plots directory exists
        os.makedirs('data_plots', exist_ok=True)

    """
    PlotGenerator.generate_mood_analysis
    ----------------------------------
    1. Create comprehensive mood analysis with multiple chart types
    """
    def generate_mood_analysis(self, mood_data, mood_counts):
        if not mood_data or len(mood_data) == 0:
            print("No mood data available for analysis")
            return
        
        try:
            fig = plt.figure(figsize=(20, 12))
            fig.suptitle('Mood Analysis Dashboard', fontsize=18, color='white')
            
            # Extract data
            timestamps = [d['timestamp'] for d in mood_data]
            moods = [d['mood'] for d in mood_data]
            confidences = [d['confidence'] for d in mood_data]
            
            # 1. Mood Timeline (Bar Chart instead of Line Chart for categorical data)
            ax1 = plt.subplot(2, 3, 1)
            # Create mood indices for plotting
            unique_moods = list(set(moods))
            mood_to_index = {mood: i for i, mood in enumerate(unique_moods)}
            mood_indices = [mood_to_index[mood] for mood in moods]
            
            ax1.scatter(timestamps, mood_indices, alpha=0.7, s=20, c='cyan')
            ax1.set_title('Mood Timeline', color='white', fontsize=14)
            ax1.set_ylabel('Mood', color='white')
            ax1.set_yticks(range(len(unique_moods)))
            ax1.set_yticklabels(unique_moods, color='white')
            ax1.tick_params(colors='white')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. Mood Distribution (Pie Chart)
            ax2 = plt.subplot(2, 3, 2)
            if mood_counts:
                mood_counts_list = list(mood_counts.values())
                mood_labels = list(mood_counts.keys())
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(mood_labels)]
                ax2.pie(mood_counts_list, labels=mood_labels, autopct='%1.1f%%', colors=colors)
                ax2.set_title('Mood Distribution', color='white', fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'No mood data', transform=ax2.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax2.set_title('Mood Distribution', color='white', fontsize=14)
            
            # 3. Confidence Over Time (Line Chart)
            ax3 = plt.subplot(2, 3, 3)
            if confidences:
                ax3.plot(timestamps, confidences, color='cyan', alpha=0.8, linewidth=2)
                ax3.set_title('Mood Confidence Over Time', color='white', fontsize=14)
                ax3.set_ylabel('Confidence', color='white')
                ax3.set_ylim(0, 1)
                ax3.tick_params(colors='white')
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No confidence data', transform=ax3.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax3.set_title('Mood Confidence Over Time', color='white', fontsize=14)
            
            # 4. Mood Frequency (Bar Chart)
            ax4 = plt.subplot(2, 3, 4)
            if mood_counts:
                mood_counts_list = list(mood_counts.values())
                mood_labels = list(mood_counts.keys())
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(mood_labels)]
                bars = ax4.bar(range(len(mood_labels)), mood_counts_list, color=colors)
                ax4.set_title('Mood Frequency', color='white', fontsize=14)
                ax4.set_ylabel('Count', color='white')
                ax4.set_xticks(range(len(mood_labels)))
                ax4.set_xticklabels(mood_labels, rotation=45)
                ax4.tick_params(colors='white')
                
                # Add value labels on bars
                for bar, count in zip(bars, mood_counts_list):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{count}', ha='center', va='bottom', color='white')
            else:
                ax4.text(0.5, 0.5, 'No mood data', transform=ax4.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax4.set_title('Mood Frequency', color='white', fontsize=14)
            
            # 5. Mood Confidence Distribution (Histogram)
            ax5 = plt.subplot(2, 3, 5)
            if confidences and len(confidences) > 1:
                ax5.hist(confidences, bins=min(20, len(confidences)//2), color='orange', alpha=0.7, edgecolor='white')
                ax5.set_title('Confidence Distribution', color='white', fontsize=14)
                ax5.set_xlabel('Confidence', color='white')
                ax5.set_ylabel('Frequency', color='white')
                ax5.tick_params(colors='white')
            else:
                ax5.text(0.5, 0.5, 'Insufficient confidence data', transform=ax5.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax5.set_title('Confidence Distribution', color='white', fontsize=14)
            
            # 6. Mood Heatmap (if enough data)
            ax6 = plt.subplot(2, 3, 6)
            if len(mood_data) > 10 and len(unique_moods) > 1:
                # Create time-based mood matrix
                mood_matrix = []
                for mood in unique_moods:
                    mood_times = [i for i, m in enumerate(moods) if m == mood]
                    mood_matrix.append(mood_times)
                
                if mood_matrix and any(len(row) > 0 for row in mood_matrix):
                    # Pad rows to same length
                    max_len = max(len(row) for row in mood_matrix)
                    padded_matrix = []
                    for row in mood_matrix:
                        padded_row = row + [None] * (max_len - len(row))
                        padded_matrix.append(padded_row)
                    
                    ax6.imshow(padded_matrix, cmap='viridis', aspect='auto')
                    ax6.set_title('Mood Over Time Heatmap', color='white', fontsize=14)
                    ax6.set_yticks(range(len(unique_moods)))
                    ax6.set_yticklabels(unique_moods, color='white')
                    ax6.set_xlabel('Time Index', color='white')
                else:
                    ax6.text(0.5, 0.5, 'Insufficient data\nfor heatmap', 
                            transform=ax6.transAxes, ha='center', va='center', 
                            color='white', fontsize=12)
                    ax6.set_title('Mood Heatmap', color='white', fontsize=14)
            else:
                ax6.text(0.5, 0.5, 'Insufficient data\nfor heatmap', 
                        transform=ax6.transAxes, ha='center', va='center', 
                        color='white', fontsize=12)
                ax6.set_title('Mood Heatmap', color='white', fontsize=14)
            
            plt.tight_layout()
            plt.savefig('data_plots/plots/mood_analysis.png', dpi=300, bbox_inches='tight', 
                       facecolor='#1a202c', edgecolor='none')
            plt.close()
            
        except Exception as e:
            print(f"Error generating mood analysis: {e}")
            plt.close('all')

    """
    PlotGenerator.generate_light_correction_analysis
    ----------------------------------------------
    1. Create light correction analysis with line charts and histograms
    """
    def generate_light_correction_analysis(self, light_data, brightness_data):
        if not light_data or len(light_data) == 0:
            print("No light correction data available for analysis")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Light Correction Analysis', fontsize=18, color='white')
            
            # Extract data
            timestamps = [d['timestamp'] for d in light_data]
            gamma_values = [d['gamma'] for d in light_data]
            
            # 1. Gamma Correction Over Time (Line Chart)
            ax1.plot(timestamps, gamma_values, color='yellow', linewidth=2, alpha=0.8)
            ax1.set_title('Gamma Correction Over Time', color='white', fontsize=14)
            ax1.set_ylabel('Gamma Value', color='white')
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Neutral (1.0)')
            ax1.legend()
            ax1.tick_params(colors='white')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. Gamma Distribution (Histogram)
            if len(gamma_values) > 1:
                ax2.hist(gamma_values, bins=min(30, len(gamma_values)//2), color='orange', alpha=0.7, edgecolor='white')
                ax2.set_title('Gamma Value Distribution', color='white', fontsize=14)
                ax2.set_xlabel('Gamma Value', color='white')
                ax2.set_ylabel('Frequency', color='white')
                ax2.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Neutral (1.0)')
                ax2.legend()
                ax2.tick_params(colors='white')
            else:
                ax2.text(0.5, 0.5, 'Insufficient gamma data', transform=ax2.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax2.set_title('Gamma Value Distribution', color='white', fontsize=14)
            
            # 3. Gamma vs Brightness Scatter (if brightness data available)
            if brightness_data and len(brightness_data) > 0:
                brightness_values = [d['brightness'] for d in brightness_data]
                # Ensure same length for scatter plot
                min_len = min(len(gamma_values), len(brightness_values))
                if min_len > 0:
                    ax3.scatter(brightness_values[:min_len], gamma_values[:min_len], alpha=0.6, color='cyan')
                    ax3.set_title('Gamma vs Brightness', color='white', fontsize=14)
                    ax3.set_xlabel('Brightness', color='white')
                    ax3.set_ylabel('Gamma Value', color='white')
                    ax3.tick_params(colors='white')
                else:
                    ax3.text(0.5, 0.5, 'No valid brightness data', transform=ax3.transAxes,
                            ha='center', va='center', color='white', fontsize=12)
                    ax3.set_title('Gamma vs Brightness', color='white', fontsize=14)
            else:
                ax3.text(0.5, 0.5, 'No brightness data available', 
                        transform=ax3.transAxes, ha='center', va='center', 
                        color='white', fontsize=12)
                ax3.set_title('Gamma vs Brightness', color='white', fontsize=14)
            
            # 4. Gamma Statistics
            if gamma_values:
                gamma_mean = np.mean(gamma_values)
                gamma_std = np.std(gamma_values)
                gamma_min = np.min(gamma_values)
                gamma_max = np.max(gamma_values)
                
                stats_text = f'Mean: {gamma_mean:.3f}\nStd: {gamma_std:.3f}\nMin: {gamma_min:.3f}\nMax: {gamma_max:.3f}'
                ax4.text(0.5, 0.5, stats_text, transform=ax4.transAxes, fontsize=14,
                        ha='center', va='center', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#2d3748', alpha=0.8))
                ax4.set_title('Gamma Statistics', color='white', fontsize=14)
            else:
                ax4.text(0.5, 0.5, 'No gamma data', transform=ax4.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax4.set_title('Gamma Statistics', color='white', fontsize=14)
            ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig('data_plots/plots/light_correction.png', dpi=300, bbox_inches='tight',
                       facecolor='#1a202c', edgecolor='none')
            plt.close()
            
        except Exception as e:
            print(f"Error generating light correction analysis: {e}")
            plt.close('all')

    """
    PlotGenerator.generate_face_detection_analysis
    --------------------------------------------
    1. Create face detection analysis with timeline and statistics
    """
    def generate_face_detection_analysis(self, face_data, session_stats):
        if not face_data or len(face_data) == 0:
            print("No face detection data available for analysis")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Face Detection Analysis', fontsize=18, color='white')
            
            # Extract data
            timestamps = [d['timestamp'] for d in face_data]
            detected = [d['detected'] for d in face_data]
            
            # 1. Face Detection Timeline (Line Chart)
            ax1.plot(timestamps, detected, color='green', linewidth=2, alpha=0.8)
            ax1.set_title('Face Detection Timeline', color='white', fontsize=14)
            ax1.set_ylabel('Face Detected', color='white')
            ax1.set_ylim(-0.1, 1.1)
            ax1.tick_params(colors='white')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. Detection Statistics
            detection_rate = session_stats.get('detection_rate', 0)
            total_frames = session_stats.get('total_frames', 0)
            faces_detected = session_stats.get('faces_detected', 0)
            
            stats_text = f'Total Frames: {total_frames:,}\nFaces Detected: {faces_detected:,}\nDetection Rate: {detection_rate:.1f}%'
            ax2.text(0.5, 0.5, stats_text, transform=ax2.transAxes, fontsize=14,
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#2d3748', alpha=0.8))
            ax2.set_title('Detection Statistics', color='white', fontsize=14)
            ax2.axis('off')
            
            # 3. Detection Rate Over Time (Rolling Average)
            if len(detected) > 10:
                window_size = min(50, len(detected) // 10)
                rolling_avg = np.convolve(detected, np.ones(window_size)/window_size, mode='valid')
                rolling_timestamps = timestamps[window_size-1:]
                
                ax3.plot(rolling_timestamps, rolling_avg, color='blue', linewidth=2, alpha=0.8)
                ax3.set_title(f'Detection Rate (Rolling Avg, Window={window_size})', color='white', fontsize=14)
                ax3.set_ylabel('Detection Rate', color='white')
                ax3.set_ylim(0, 1)
                ax3.tick_params(colors='white')
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax3.text(0.5, 0.5, 'Insufficient data\nfor rolling average', 
                        transform=ax3.transAxes, ha='center', va='center', 
                        color='white', fontsize=12)
                ax3.set_title('Detection Rate Over Time', color='white', fontsize=14)
            
            # 4. Detection Pattern (Bar Chart)
            # Count consecutive detections/non-detections
            if detected:
                detection_changes = []
                current_state = detected[0]
                current_count = 1
                
                for d in detected[1:]:
                    if d == current_state:
                        current_count += 1
                    else:
                        detection_changes.append((current_state, current_count))
                        current_state = d
                        current_count = 1
                detection_changes.append((current_state, current_count))
                
                if detection_changes:
                    states = ['Detected' if state else 'Not Detected' for state, _ in detection_changes]
                    counts = [count for _, count in detection_changes]
                    colors = ['green' if state else 'red' for state, _ in detection_changes]
                    
                    bars = ax4.bar(range(len(states)), counts, color=colors, alpha=0.7)
                    ax4.set_title('Detection Pattern', color='white', fontsize=14)
                    ax4.set_ylabel('Duration (frames)', color='white')
                    ax4.set_xticks(range(len(states)))
                    ax4.set_xticklabels(states, rotation=45)
                    ax4.tick_params(colors='white')
                    
                    # Add value labels
                    for bar, count in zip(bars, counts):
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{count}', ha='center', va='bottom', color='white')
                else:
                    ax4.text(0.5, 0.5, 'No detection changes', transform=ax4.transAxes,
                            ha='center', va='center', color='white', fontsize=12)
                    ax4.set_title('Detection Pattern', color='white', fontsize=14)
            else:
                ax4.text(0.5, 0.5, 'No detection data', transform=ax4.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax4.set_title('Detection Pattern', color='white', fontsize=14)
            
            plt.tight_layout()
            plt.savefig('data_plots/plots/face_detection.png', dpi=300, bbox_inches='tight',
                       facecolor='#1a202c', edgecolor='none')
            plt.close()
            
        except Exception as e:
            print(f"Error generating face detection analysis: {e}")
            plt.close('all')

    """
    PlotGenerator.generate_system_metrics
    -----------------------------------
    1. Create comprehensive system metrics dashboard
    """
    def generate_system_metrics(self, mic_data, privacy_data, fps_data, session_stats):
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('System Metrics Dashboard', fontsize=18, color='white')
            
            # 1. Session Information
            session_duration = session_stats.get('duration_seconds', 0)
            duration_text = f'Session Duration:\n{timedelta(seconds=int(session_duration))}\n\nAverage FPS:\n{session_stats.get("average_fps", 0):.1f}'
            ax1.text(0.5, 0.5, duration_text, transform=ax1.transAxes, fontsize=12,
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#2d3748', alpha=0.8))
            ax1.set_title('Session Info', color='white', fontsize=14)
            ax1.axis('off')
            
            # 2. Privacy Mode Usage (Pie Chart)
            privacy_usage = session_stats.get('privacy_mode_usage', {})
            if privacy_usage and any(privacy_usage.values()):
                modes = list(privacy_usage.keys())
                counts = list(privacy_usage.values())
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(modes)]
                ax2.pie(counts, labels=modes, autopct='%1.1f%%', colors=colors)
                ax2.set_title('Privacy Mode Usage', color='white', fontsize=14)
            else:
                ax2.text(0.5, 0.5, 'No privacy mode data', transform=ax2.transAxes,
                        ha='center', va='center', color='white')
                ax2.set_title('Privacy Mode Usage', color='white', fontsize=14)
                ax2.axis('off')
            
            # 3. Microphone Status Over Time (Line Chart)
            if mic_data and len(mic_data) > 0:
                timestamps = [d['timestamp'] for d in mic_data]
                muted = [d['muted'] for d in mic_data]
                ax3.plot(timestamps, muted, color='red', linewidth=2, alpha=0.8)
                ax3.set_title('Microphone Status', color='white', fontsize=14)
                ax3.set_ylabel('Muted', color='white')
                ax3.set_ylim(-0.1, 1.1)
                ax3.tick_params(colors='white')
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No mic status data', transform=ax3.transAxes,
                        ha='center', va='center', color='white')
                ax3.set_title('Microphone Status', color='white', fontsize=14)
                ax3.axis('off')
            
            # 4. Performance Metrics
            perf_text = f'Detection Rate: {session_stats.get("detection_rate", 0):.1f}%\nTotal Moods: {sum(session_stats.get("mood_counts", {}).values())}\nMic Mutes: {session_stats.get("mic_mute_count", 0)}\nMic Unmutes: {session_stats.get("mic_unmute_count", 0)}'
            ax4.text(0.5, 0.5, perf_text, transform=ax4.transAxes, fontsize=12,
                    ha='center', va='center', color='white',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#2d3748', alpha=0.8))
            ax4.set_title('Performance Metrics', color='white', fontsize=14)
            ax4.axis('off')
            
            plt.tight_layout()
            plt.savefig('data_plots/plots/system_metrics.png', dpi=300, bbox_inches='tight',
                       facecolor='#1a202c', edgecolor='none')
            plt.close()
            
        except Exception as e:
            print(f"Error generating system metrics: {e}")
            plt.close('all')

    """
    PlotGenerator.generate_performance_analysis
    -----------------------------------------
    1. Create performance analysis with FPS tracking and optimization insights
    """
    def generate_performance_analysis(self, fps_data, session_stats):
        if not fps_data or len(fps_data) == 0:
            print("No FPS data available for performance analysis")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Performance Analysis', fontsize=18, color='white')
            
            # Extract FPS data
            timestamps = [d['timestamp'] for d in fps_data]
            fps_values = [d['fps'] for d in fps_data]
            
            # 1. FPS Over Time (Line Chart)
            ax1.plot(timestamps, fps_values, color='cyan', linewidth=2, alpha=0.8)
            ax1.set_title('FPS Over Time', color='white', fontsize=14)
            ax1.set_ylabel('FPS', color='white')
            ax1.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Target (30 FPS)')
            ax1.legend()
            ax1.tick_params(colors='white')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
            
            # 2. FPS Distribution (Histogram)
            if len(fps_values) > 1:
                ax2.hist(fps_values, bins=min(20, len(fps_values)//2), color='purple', alpha=0.7, edgecolor='white')
                ax2.set_title('FPS Distribution', color='white', fontsize=14)
                ax2.set_xlabel('FPS', color='white')
                ax2.set_ylabel('Frequency', color='white')
                ax2.axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Target (30 FPS)')
                ax2.legend()
                ax2.tick_params(colors='white')
            else:
                ax2.text(0.5, 0.5, 'Insufficient FPS data', transform=ax2.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax2.set_title('FPS Distribution', color='white', fontsize=14)
            
            # 3. Performance Statistics
            if fps_values:
                fps_mean = np.mean(fps_values)
                fps_std = np.std(fps_values)
                fps_min = np.min(fps_values)
                fps_max = np.max(fps_values)
                
                perf_text = f'Average FPS: {fps_mean:.1f}\nStd Dev: {fps_std:.1f}\nMin FPS: {fps_min:.1f}\nMax FPS: {fps_max:.1f}\n\nTarget: 30 FPS\nPerformance: {"Good" if fps_mean >= 25 else "Needs Optimization"}'
                ax3.text(0.5, 0.5, perf_text, transform=ax3.transAxes, fontsize=12,
                        ha='center', va='center', color='white',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='#2d3748', alpha=0.8))
                ax3.set_title('Performance Statistics', color='white', fontsize=14)
            else:
                ax3.text(0.5, 0.5, 'No FPS data', transform=ax3.transAxes,
                        ha='center', va='center', color='white', fontsize=12)
                ax3.set_title('Performance Statistics', color='white', fontsize=14)
            ax3.axis('off')
            
            # 4. Performance Trend (Rolling Average)
            if len(fps_values) > 10:
                window_size = min(20, len(fps_values) // 5)
                rolling_avg = np.convolve(fps_values, np.ones(window_size)/window_size, mode='valid')
                rolling_timestamps = timestamps[window_size-1:]
                
                ax4.plot(rolling_timestamps, rolling_avg, color='green', linewidth=2, alpha=0.8)
                ax4.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Target (30 FPS)')
                ax4.set_title(f'FPS Trend (Rolling Avg, Window={window_size})', color='white', fontsize=14)
                ax4.set_ylabel('FPS', color='white')
                ax4.legend()
                ax4.tick_params(colors='white')
                ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data\nfor trend analysis', 
                        transform=ax4.transAxes, ha='center', va='center', 
                        color='white', fontsize=12)
                ax4.set_title('FPS Trend', color='white', fontsize=14)
            
            plt.tight_layout()
            plt.savefig('data_plots/plots/performance_analysis.png', dpi=300, bbox_inches='tight',
                       facecolor='#1a202c', edgecolor='none')
            plt.close()
            
        except Exception as e:
            print(f"Error generating performance analysis: {e}")
            plt.close('all')

    """
    PlotGenerator.generate_all_plots
    -------------------------------
    1. Generate all analysis plots using the data tracker
    """
    def generate_all_plots(self, data_tracker):
        print("Generating comprehensive data analysis plots...")
        
        try:
            # Get session statistics
            session_stats = data_tracker.get_session_stats()
            
            # Generate all plots
            self.generate_mood_analysis(data_tracker.mood_data, data_tracker.mood_counts)
            self.generate_light_correction_analysis(data_tracker.light_correction_data, data_tracker.brightness_data)
            self.generate_face_detection_analysis(data_tracker.face_detection_data, session_stats)
            self.generate_system_metrics(data_tracker.mic_status_data, data_tracker.privacy_mode_data, 
                                       data_tracker.fps_data, session_stats)
            self.generate_performance_analysis(data_tracker.fps_data, session_stats)
            
            # Save session data
            data_tracker.save_session_data()
            
            print("Data analysis complete! Check the 'data_plots' folder for generated PNG files.")
            
        except Exception as e:
            print(f"Error generating all plots: {e}")
            plt.close('all')