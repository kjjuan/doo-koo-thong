FROM python:3.11

#Setting the working directory
WORKDIR /app

#Copy requirements file into the container
COPY requirements.txt .

#Installing dependencies
RUN pip install -r requirements.txt

#Copy the entire kedro project into the container
COPY . .

CMD ["python", "-m", "kedro", "run"]