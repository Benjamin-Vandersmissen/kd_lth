eval "$(ssh-agent -s)"  &> /dev/null
ssh-add -q /project_antwerp/github_ssh/gpulab_github

ssh-keyscan github.com >> ~/.ssh/known_hosts  # This is unsecure!!! We might need to use a secure approach like https://serverfault.com/questions/856194/securely-add-a-host-e-g-github-to-the-ssh-known-hosts-file/1098531#1098531


if [[ $(git status --porcelain) ]]; then
    # Just in case we do not have the correct branches yet.
    git fetch --all
    git checkout gpulab
    
    git config --global user.name "AutoUpdater (via GPULab)"
    git config --global user.email "benjaminv55@gmail.com"
    git add -A .
    git commit -m "[$(date)] Automatic backup of GPULab code. 
Triggered by job $GPULAB_JOB_ID of project $GPULAB_PROJECT_NAME." 
    git push origin gpulab
else
    echo "No updated files"
fi
ssh-agent -k &> /dev/null