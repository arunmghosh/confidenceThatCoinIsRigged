import os
import sys

def main():
    print("=" * 60)
    print("INVESTIGATION: AI CONFIDENCE IN RIGGED COIN DETECTION")
    print("=" * 60)

    if not os.path.exists("checkpoints/model_a.pth"):
        print("\nStep 1: Training Models A, B, and C with Brier Score Loss...")
        import train
        train.train()
    else:
        print("\nStep 1: Found existing model checkpoints in checkpoints/.")

    print("\nStep 2: Evaluating Models across experimental grid...")
    import evaluate
    evaluate.evaluate_models(num_trials=500)

    print("\nStep 3: Generating Publication Figures...")
    import visualize
    visualize.generate_all_plots()

    print("\nStep 4: Generating Interactive HTML Dashboard...")
    import generate_report
    generate_report.create_html_dashboard()

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS & ARTIFACTS SUCCESSFULLY GENERATED!")
    print("  - Results: results/evaluation_summary.csv")
    print("  - Figures: figures/*.png")
    print("  - Dashboard: dashboard.html")
    print("=" * 60)

if __name__ == "__main__":
    main()
