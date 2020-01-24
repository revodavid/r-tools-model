## Test run using Accidents data
## from: https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-r-experiment

library(azuremlsdk)

ws <- load_workspace_from_config()

experiment_name <- "accident-logreg"
exp <- experiment(ws, experiment_name)

cluster_name <- "rcluster"
compute_target <- get_compute(ws, cluster_name = cluster_name)
if (is.null(compute_target)) {
  vm_size <- "STANDARD_D2_V2" 
  compute_target <- create_aml_compute(workspace = ws,
                                       cluster_name = cluster_name,
                                       vm_size = vm_size,
                                       max_nodes = 1)
  
  wait_for_provisioning_completion(compute_target, show_output = TRUE)
}

nassCDS <- read.csv("nassCDS.csv", 
                    colClasses=c("factor","numeric","factor",
                                 "factor","factor","numeric",
                                 "factor","numeric","numeric",
                                 "numeric","character","character",
                                 "numeric","numeric","character"))

accidents <- na.omit(nassCDS[,c("dead","dvcat","seatbelt","frontal","sex","ageOFocc","yearVeh","airbag","occRole")])
accidents$frontal <- factor(accidents$frontal, labels=c("notfrontal","frontal"))
accidents$occRole <- factor(accidents$occRole)
accidents$dvcat <- ordered(accidents$dvcat, 
                           levels=c("1-9km/h","10-24","25-39","40-54","55+"))

saveRDS(accidents, file="accidents.Rd")

ds <- get_default_datastore(ws)

target_path <- "accidentdata"
upload_files_to_datastore(ds,
                          list("./accidents.Rd"),
                          target_path = target_path,
                          overwrite = TRUE)

est <- estimator(source_directory = ".",
                 entry_script = "accidents.R",
                 script_params = list("--data_folder" = ds$path(target_path)),
                 compute_target = compute_target)

run <- submit_experiment(exp, est)
view_run_details(run)

metrics <- get_run_metrics(run)
metrics

download_files_from_run(run, prefix="outputs/")
accident_model <- readRDS("outputs/model.rds")
summary(accident_model)

newdata <- data.frame( # valid values shown below
  dvcat="10-24",        # "1-9km/h" "10-24"   "25-39"   "40-54"   "55+"  
  seatbelt="none",      # "none"   "belted"  
  frontal="frontal",    # "notfrontal" "frontal"
  sex="f",              # "f" "m"
  ageOFocc=16,          # age in years, 16-97
  yearVeh=2002,         # year of vehicle, 1955-2003
  airbag="none",        # "none"   "airbag"   
  occRole="pass"        # "driver" "pass"
)

## predicted probability of death for these variables, as a percentage
as.numeric(predict(accident_model,newdata, type="response")*100)

model <- register_model(ws, 
                        model_path = "outputs/model.rds", 
                        model_name = "accidents_model",
                        description = "Predict probablity of auto accident")

r_env <- r_environment(name = "basic_env")

inference_config <- inference_config(
  entry_script = "accidents_predict.R",
  source_directory = ".",
  environment = r_env)

aci_config <- aci_webservice_deployment_config(cpu_cores = 1, memory_gb = 0.5)
aci_service <- deploy_model(ws, 
                            'accident-pred', 
                            list(model), 
                            inference_config, 
                            aci_config)

wait_for_deployment(aci_service, show_output = TRUE)

library(jsonlite)

newdata <- data.frame( # valid values shown below
  dvcat="10-24",        # "1-9km/h" "10-24"   "25-39"   "40-54"   "55+"  
  seatbelt="none",      # "none"   "belted"  
  frontal="frontal",    # "notfrontal" "frontal"
  sex="f",              # "f" "m"
  ageOFocc=22,          # age in years, 16-97
  yearVeh=2002,         # year of vehicle, 1955-2003
  airbag="none",        # "none"   "airbag"   
  occRole="pass"        # "driver" "pass"
)

prob <- invoke_webservice(aci_service, toJSON(newdata))
prob