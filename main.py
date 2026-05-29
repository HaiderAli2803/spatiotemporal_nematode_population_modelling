import sys

def main():
    print("=" * 60)
    print("SPATIOTEMPORAL NEMATODE POPULATION MODELLING SIMULATION")
    print("=" * 60)
    print("1. Run Baseline GMM (baseline_gmm.py)")
    print("2. Run Risk Surface Assessment (risk_surface_assessment.py)")
    print("3. Run Full Pipeline Handoff Demo (pipeline_handoff.py)")
    print("4. Run Sampling Efficiency Analysis (sampling_efficiency.py)")
    print("5. Run Parameter Sensitivity Analysis (sensitivity_analysis.py)")
    print("6. Run Visualisations (visualisations.py)")
    print("7. Exit")
    print("-" * 60)
    
    choice = input("Select a simulation module to execute (1-7): ").strip()
    
    print("\nExecuting module...\n")
    
    try:
        if choice == '1':
            import baseline_gmm
        elif choice == '2':
            import risk_surface_assessment
        elif choice == '3':
            import pipeline_handoff
        elif choice == '4':
            import sampling_efficiency
        elif choice == '5':
            import sensitivity_analysis
        elif choice == '6':
            import visualisations
        elif choice == '7':
            print("Exiting simulation platform.")
            sys.exit()
        else:
            print("Invalid. Please run the script again and choose 1-7.")
    except ImportError as e:
        print(f"Error: Could not import the module. Incorrect file name.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()
