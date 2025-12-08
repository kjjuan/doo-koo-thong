# done by matthew

#Define the pipeline to run upon running run.sh
PIPELINE_NAME="${1:-__default__}"

#For any extra arguements to run specific pipelines or nodes
EXTRA_ARGS="${@:2}"

#Running the command
kedro run 

#To show whether run.sh is running
echo "Running Kedro pipeline: $PIPELINE_NAME"

#Check whether the pipeline was excecuted properly
if [ $? -eq 0 ]; then
  echo "✅ Pipeline '$PIPELINE_NAME' completed successfully."
else
  echo "❌ Pipeline '$PIPELINE_NAME' failed."
  exit 1
fi