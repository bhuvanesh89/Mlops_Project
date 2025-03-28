create env

conda create -n wineq python=3.7 -y
activate env

conda activate wineq
created a req file

install the req

pip install -r requirements.txt
download the data from

https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing

git init

dvc init

dvc add data_given/winequality.csv

git add .

git commit -m "first commit"

git add . && git commit -m "update Readme.md"

git remote add origin https://github.com/PavanTeja5/Mlops_Project.git

git branch -M main

git push -u origin main 

tox command

tox

for rebuilding -

tox -r

pytest -v

setup commands -
pip install -e .

build your own package command-
python setup.py sdist bdist_wheel


Final Output

https://wine-quality-prediction-3e08.onrender.com

create an artifacts folder
mkdir artifacts

mlflow server command -
  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 127.0.0.1 --port 1234

dvc repro
