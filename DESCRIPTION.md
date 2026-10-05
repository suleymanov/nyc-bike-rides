#### Quickest demo

1) Create Docker + FastAPI solution to serve a simplest viable demo (inference model)

2) Deploy Gitlab CI locally and serve a simplest viable demo

3) Deploy simplest viable demo on gitlab.com

4) Deploy simplest viable demo with Github Actions



#### Full project outline (MLOps Zoomcamp course)

URL: https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/07-project

1) Problem description - should be well described and it should be clear what problem the project solves

2) Cloud - the project is developed on the cloud and Terraform is used to provision the infrastructure

3) Experiment tracking and model registry are used

4) Workflow orchestration - fully deployed

5) Model deployment - model deployment code is containerized and can be deployed to cloud

6) Model monitoring - comprehensive model monitoring that sends alerts or runs a conditional workflow (retraining, generating debugging dashboard, switch to different model) if metrics threshold is violated

7) Reproducibility - clear instructions on how to run the code; dependencies are specified

8) Best practices
	
	- unit tests

	- integration test

	- linter and code formatter are used

	- makefile exists

	- pre-commit hooks

	- CI/CD pipeline
