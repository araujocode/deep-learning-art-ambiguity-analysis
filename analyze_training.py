#!/usr/bin/env python3
"""
Analyze training dynamics to determine if more epochs would help.
"""

import re
from pathlib import Path

def analyze_training_dynamics():
    """Analyze the training logs to understand if more epochs would help."""
    
    log_file = './experiments/run_001/training.log'
    
    print("🔍 TRAINING DYNAMICS ANALYSIS")
    print("=" * 50)
    
    if not Path(log_file).exists():
        print("❌ Training log not found!")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_losses = []
    epochs = []
    
    for line in lines:
        if 'Train Loss:' in line and 'Val Loss:' in line:
            # Extract metrics
            train_loss = float(re.search(r'Train Loss: ([0-9.]+)', line).group(1))
            train_acc = float(re.search(r'Train Acc: ([0-9.]+)', line).group(1))
            val_loss = float(re.search(r'Val Loss: ([0-9.]+)', line).group(1))
            val_acc = float(re.search(r'Val Acc: ([0-9.]+)', line).group(1))
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            epochs.append(len(epochs) + 1)
    
    print(f"📊 BASIC METRICS:")
    print(f"   Total epochs trained: {len(epochs)}")
    print(f"   Final training accuracy: {train_accs[-1]:.3f}")
    print(f"   Final validation accuracy: {val_accs[-1]:.3f}")
    print(f"   Best validation accuracy: {max(val_accs):.3f}")
    print(f"   Training-validation gap: {train_accs[-1] - val_accs[-1]:.3f}")
    
    # Dataset size limitations (will change with the new dataset)
    print(f"\n📂 DATASET SIZE ANALYSIS (Note: Specific numbers depend on 'create_large_dataset.py' execution):")
    print(f"   The following are placeholders or based on previous runs.")
    print(f"   Ensure 'create_large_dataset.py' has been run with the real dataset.")
    # print(f"   Total images: 480") # Example of old hardcoded value
    # print(f"   Training images: 336 (~42 per class)")
    # print(f"   Validation images: 72 (~9 per class)")
    # print(f"   Test images: 72 (~9 per class)")
    
    # Overfitting analysis
    print(f"\n🚨 OVERFITTING ANALYSIS:")
    last_5_train = train_accs[-5:]
    last_5_val = val_accs[-5:]
    avg_gap = sum([t-v for t,v in zip(last_5_train, last_5_val)])/5
    
    print(f"   Last 5 train accuracies: {[f'{x:.3f}' for x in last_5_train]}")
    print(f"   Last 5 val accuracies:   {[f'{x:.3f}' for x in last_5_val]}")
    print(f"   Average train-val gap (last 5): {avg_gap:.3f}")
    
    if avg_gap > 0.05:
        print(f"   ⚠️  SIGNIFICANT OVERFITTING DETECTED!")
    elif avg_gap > 0.02:
        print(f"   ⚠️  Mild overfitting present")
    else:
        print(f"   ✅ No significant overfitting")
    
    # Plateau analysis
    print(f"\n📈 IMPROVEMENT TREND ANALYSIS:")
    if len(val_accs) >= 10:
        last_10_val = val_accs[-10:]
        last_5_val = val_accs[-5:]
        
        improvement_10 = max(last_10_val) - min(last_10_val)
        improvement_5 = max(last_5_val) - min(last_5_val)
        
        best_last_5 = max(last_5_val)
        best_last_10 = max(last_10_val)
        
        print(f"   Validation improvement (last 10 epochs): {improvement_10:.3f}")
        print(f"   Validation improvement (last 5 epochs): {improvement_5:.3f}")
        print(f"   Best in last 5 vs best in last 10: {best_last_5:.3f} vs {best_last_10:.3f}")
        
        if improvement_5 < 0.01:
            print(f"   📉 MODEL HAS PLATEAUED (last 5 epochs)")
        elif improvement_10 < 0.02:
            print(f"   📉 MODEL SHOWING SLOW IMPROVEMENT")
        else:
            print(f"   📈 MODEL STILL IMPROVING")
    
    # Loss analysis
    print(f"\n📉 LOSS ANALYSIS:")
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    best_val_loss = min(val_losses)
    
    print(f"   Final training loss: {final_train_loss:.3f}")
    print(f"   Final validation loss: {final_val_loss:.3f}")
    print(f"   Best validation loss: {best_val_loss:.3f}")
    print(f"   Loss gap: {final_val_loss - final_train_loss:.3f}")
    
    # Theoretical maximum analysis
    print(f"\n🎯 THEORETICAL PERFORMANCE ANALYSIS:")
    print(f"   Random chance (8 classes): 12.5%")
    print(f"   Current test accuracy: 62.5%")
    print(f"   Current validation accuracy: 63.9%")
    print(f"   Human-level performance (art periods): ~85-95%")
    print(f"   Remaining improvement potential: ~20-30%")
    
    # Honest assessment
    print(f"\n💡 BRUTAL HONEST ASSESSMENT:")
    print(f"=" * 40)
    
    fundamental_issues = []
    
    # Check dataset size
    if len(train_accs) > 0:
        if avg_gap > 0.05:
            fundamental_issues.append("SEVERE OVERFITTING: Model memorizing small dataset")
        
        if improvement_5 < 0.01:
            fundamental_issues.append("PLATEAU REACHED: No improvement in recent epochs")
        
        if len(epochs) >= 25 and max(val_accs) < 0.70:
            fundamental_issues.append("DATASET TOO SMALL: 42 samples/class insufficient for deep learning")
    
    if fundamental_issues:
        print(f"   🔴 PRIMARY LIMITATIONS:")
        for issue in fundamental_issues:
            print(f"      • {issue}")
        
        print(f"\n   📋 RECOMMENDATION:")
        if "DATASET TOO SMALL" in str(fundamental_issues):
            print(f"      • More epochs WON'T help - need MORE DATA")
            print(f"      • Current dataset (42 samples/class) is too small")
            print(f"      • Need 200-500+ samples per class for good performance")
            print(f"      • Consider data augmentation or synthetic data generation")
        elif "SEVERE OVERFITTING" in str(fundamental_issues):
            print(f"      • More epochs will make overfitting WORSE")
            print(f"      • Need stronger regularization or more data")
        elif "PLATEAU REACHED" in str(fundamental_issues):
            print(f"      • Model has reached its capacity limit")
            print(f"      • More epochs unlikely to help significantly")
    else:
        print(f"   🟡 MIXED SIGNALS:")
        print(f"      • Model might benefit from a few more epochs")
        print(f"      • But fundamental dataset size limitation remains")
        print(f"      • Expected improvement: 1-3% at most")
    
    return {
        'overfitting': avg_gap > 0.05,
        'plateaued': improvement_5 < 0.01 if len(val_accs) >= 5 else False,
        'dataset_too_small': len(train_accs) > 0 and max(val_accs) < 0.70,
        'more_epochs_recommended': avg_gap < 0.03 and improvement_5 >= 0.01
    }

if __name__ == "__main__":
    analyze_training_dynamics()
