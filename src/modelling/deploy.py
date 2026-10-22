# Deployment: retrains the model every Monday at 3am. Needs the prefect server to be running.

from src.modelling.main import main

if __name__ == "__main__":
    # serve = register the flow as a deployment and keep this process running to launch the runs
    main.serve(
        name="weekly-retraining",
        # cron format: minute hour day-of-month month day-of-week -> "0 3 * * 1" = Monday 3:00
        cron="0 3 * * 1",
        # The argument given to main() at each run
        parameters={"trainset_path": "data/abalone.csv"},
    )
