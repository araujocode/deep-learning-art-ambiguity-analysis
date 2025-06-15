#!/usr/bin/env python3
"""
Analyze training dynamics to determine if more epochs would help.
"""

import re
import argparse # Added
from pathlib import Path
import numpy as np # Added for np.mean

def analyze_training_dynamics(experiment_dir: str): # Added argument
    """Analyze the training logs to understand if more epochs would help."""
    
    log_file = Path(experiment_dir) / 'training.log' # Made path dynamic
    
    print("\\n" + "=" * 50) # Added newline for better separation
    print("🔍 TRAINING DYNAMICS ANALYSIS")
    print("=" * 50)
    
    if not log_file.exists():
        print(f"❌ Training log not found at {log_file}!") # Updated path in message
        return None # Return None if log not found
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    epochs_list = [] # Renamed from epochs to avoid conflict
    
    for line in lines:
        if 'Train Loss:' in line and 'Val Loss:' in line:
            try: # Added try-except for robust parsing
                train_loss = float(re.search(r'Train Loss: ([0-9.]+)', line).group(1))
                train_acc = float(re.search(r'Train Acc: ([0-9.]+)', line).group(1))
                val_loss = float(re.search(r'Val Loss: ([0-9.]+)', line).group(1))
                val_acc = float(re.search(r'Val Acc: ([0-9.]+)', line).group(1))
                
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                epochs_list.append(len(epochs_list) + 1)
            except AttributeError:
                print(f"Warning: Could not parse metrics from line: {line.strip()}")
                continue # Skip malformed lines

    if not val_accs: # Check if any metrics were actually parsed
        print("❌ No valid epoch data found in the log file.")
        return None

    print(f"📊 BASIC METRICS (from {log_file}):") # Clarified source
    print(f"   Total epochs trained: {len(epochs_list)}")
    print(f"   Final training accuracy: {train_accs[-1]:.4f}") # Increased precision
    print(f"   Final validation accuracy: {val_accs[-1]:.4f}") # Increased precision
    print(f"   Best validation accuracy: {max(val_accs):.4f} (at epoch {np.argmax(val_accs) + 1})") # Added epoch of best val_acc
    print(f"   Training-validation accuracy gap (final): {train_accs[-1] - val_accs[-1]:.4f}") # Increased precision
    
    # Removed outdated DATASET SIZE ANALYSIS section
    
    # Overfitting analysis
    print(f"\\n🚨 OVERFITTING ANALYSIS:")
    if len(train_accs) >= 5:
        last_5_train_acc = train_accs[-5:]
        last_5_val_acc = val_accs[-5:]
        avg_acc_gap_last_5 = np.mean([t - v for t, v in zip(last_5_train_acc, last_5_val_acc)])
        
        print(f"   Last 5 train accuracies: {[f'{x:.4f}' for x in last_5_train_acc]}")
        print(f"   Last 5 val accuracies:   {[f'{x:.4f}' for x in last_5_val_acc]}")
        print(f"   Average train-val accuracy gap (last 5 epochs): {avg_acc_gap_last_5:.4f}")
        
        if avg_acc_gap_last_5 > 0.10: # Adjusted threshold
            print(f"   ⚠️  SIGNIFICANT OVERFITTING DETECTED (gap > 0.10)!")
        elif avg_acc_gap_last_5 > 0.05: # Adjusted threshold
            print(f"   ⚠️  Mild overfitting present (gap > 0.05)")
        else:
            print(f"   ✅ No significant overfitting detected (gap <= 0.05)")
    else:
        print("   Not enough epochs to perform last 5 epoch overfitting analysis.")

    # Plateau analysis for validation accuracy
    print(f"\\n📈 VALIDATION ACCURACY TREND ANALYSIS:")
    improvement_threshold = 0.005 # Stricter threshold for plateau
    if len(val_accs) >= 10:
        # improvement_10_epochs = val_accs[-1] - val_accs[-10] # Improvement over last 10
        improvement_5_epochs = val_accs[-1] - val_accs[-5]   # Improvement over last 5
        # improvement_abs_5_epochs = max(val_accs[-5:]) - min(val_accs[-5:]) # Absolute fluctuation
        
        # print(f"   Validation accuracy change (last 10 epochs): {improvement_10_epochs:+.4f}")
        print(f"   Validation accuracy change (last 5 epochs): {improvement_5_epochs:+.4f}")
        # print(f"   Validation accuracy fluctuation (range in last 5 epochs): {improvement_abs_5_epochs:.4f}")

        if improvement_5_epochs < improvement_threshold and (max(val_accs[-5:]) - np.mean(val_accs[-5:]) < improvement_threshold):
             print(f"   📉 MODEL VALIDATION ACCURACY HAS LIKELY PLATEAUED (improvement < {improvement_threshold} in last 5 epochs and stable).")
        elif improvement_5_epochs < improvement_threshold * 2 : # Slightly less strict
             print(f"   📉 MODEL VALIDATION ACCURACY SHOWING SLOW IMPROVEMENT.")
        else:
             print(f"   📈 MODEL VALIDATION ACCURACY STILL IMPROVING.")
    elif len(val_accs) >= 5:
        improvement_5_epochs = val_accs[-1] - val_accs[-5]
        print(f"   Validation accuracy change (last 5 epochs): {improvement_5_epochs:+.4f}")
        if improvement_5_epochs < improvement_threshold:
            print(f"   📉 MODEL VALIDATION ACCURACY HAS LIKELY PLATEAUED (improvement < {improvement_threshold} in last 5 epochs).")
        else:
            print(f"   📈 MODEL VALIDATION ACCURACY STILL IMPROVING (based on last 5 epochs).")
    else:
        print("   Not enough epochs for detailed trend analysis.")
        
    # Loss analysis
    print(f"\\n📉 LOSS ANALYSIS:")
    if train_losses and val_losses:
        final_train_loss = train_losses[-1]
        final_val_loss = val_losses[-1]
        best_val_loss = min(val_losses)
        epoch_best_val_loss = np.argmin(val_losses) + 1
        
        print(f"   Final training loss: {final_train_loss:.4f}")
        print(f"   Final validation loss: {final_val_loss:.4f}")
        print(f"   Best validation loss: {best_val_loss:.4f} (at epoch {epoch_best_val_loss})")
        print(f"   Validation loss gap (final_val - final_train): {final_val_loss - final_train_loss:.4f}")
        if final_val_loss > best_val_loss * 1.1 and len(val_losses) - epoch_best_val_loss > 3 : # If current val_loss is 10% worse than best and it's been a few epochs
            print(f"   ⚠️  Validation loss started increasing after epoch {epoch_best_val_loss}. Consider early stopping or reducing epochs.")

    # Removed THEORETICAL PERFORMANCE ANALYSIS and BRUTAL HONEST ASSESSMENT sections
    # as they contained many assumptions and hardcoded values.
    # The analysis above provides quantitative data for the user to interpret.

    print("\\n" + "=" * 50)
    print("END OF TRAINING DYNAMICS ANALYSIS")
    print("=" * 50 + "\\n")
    
    # Return a dictionary of key metrics, can be expanded
    analysis_summary = {}
    if val_accs:
        analysis_summary['best_val_acc'] = max(val_accs)
        analysis_summary['best_val_acc_epoch'] = np.argmax(val_accs) + 1
        analysis_summary['final_val_acc'] = val_accs[-1]
        if len(val_accs) >= 5:
            analysis_summary['val_acc_plateaued'] = (val_accs[-1] - val_accs[-5]) < improvement_threshold
    if val_losses:
        analysis_summary['best_val_loss'] = min(val_losses)
        analysis_summary['best_val_loss_epoch'] = np.argmin(val_losses) + 1
    
    return analysis_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze training dynamics from a log file.") # Added parser
    parser.add_argument("--experiment_dir", type=str, required=True, # Added argument
                       help="Path to the experiment directory containing training.log")
    args = parser.parse_args()
    
    analyze_training_dynamics(args.experiment_dir)
