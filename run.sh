# git pull

# Delete all `.DS_Store`
find . -name ".DS_Store" -delete

# De;ete all `.ipynb_checkpoints` folders
find . -name .ipynb_checkpoints -type d -exec rm -rf {} \;

# jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute --inplace *.ipynb
# jupyter nbconvert --to script *.ipynb

ipython COVID-19.py

ipython Florida\ R_0.py

black *.py

git add --all && git commit -m "Update" && git push
