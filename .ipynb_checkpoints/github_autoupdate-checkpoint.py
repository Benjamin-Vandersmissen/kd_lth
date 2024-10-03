import os
import subprocess
import re
from datetime import datetime

class GithubAutoUpdater: 
    def __init__(self, private_key='/project_antwerp/github_ssh/gpulab_github'):
        self.private_key = private_key
        self.env = os.environ
        self.env.update(self.initialize_ssh_agent())
        self.add_ssh_key()
        
    def initialize_ssh_agent(self):
        output = subprocess.check_output(['ssh-agent', '-s'], shell=True)
        result = {}
        for name, value in re.findall(r'([A-Z_]+)=([^;]+);',
                                      output.decode('ascii')):
            result[name] = value
        return result
    
    def add_ssh_key(self):
        subprocess.check_call(['ssh-add', self.private_key], env=self.env)
    
    @staticmethod
    def github_is_known_host():
        ssh_path = os.path.expanduser('~/.ssh/known_hosts')
        if os.path.exists(ssh_path):
            with open(ssh_path) as f:
                lines = f.readlines()
            for line in lines:
                if 'github' in line:
                    return True
        return False
            
    @staticmethod
    def add_github_to_known_hosts():
        os.makedirs(os.path.expanduser('~/.ssh'), exist_ok=True)
        subprocess.run(['ssh-keyscan github.com >> ~/.ssh/known_hosts'], shell=True)
        print("GITHUB added to known hosts.")

    def github_needs_updating(self):
        cur_directory = os.path.dirname(os.path.realpath(__file__))
        subprocess.run([f'git config --global --add safe.directory {cur_directory}'], shell=True, env=self.env)  # Avoids dubious ownership errors
        result = subprocess.run(['git status --porcelain'], shell=True, capture_output=True, env=self.env)
        return len(result.stdout) != 0
    
    def update_github(self):
        subprocess.run(['git fetch --all'], shell=True, env=self.env, capture_output=True)
        subprocess.run(['git checkout gpulab'], shell=True, env=self.env, capture_output=True)
        subprocess.run(['git config --global user.name "AutoUpdater (via GPULab)"'], shell=True, env=self.env, capture_output=True)
        subprocess.run(['git config --global user.email "benjaminv55@gmail.com"'], shell=True, env=self.env, capture_output=True)
        subprocess.run(['git add -A'], shell=True, env=self.env, capture_output=True)
        subprocess.run([f'git commit -m "[{datetime.now()}] Automatic backup of GPULab code triggered by job {self.env["GPULAB_JOB_ID"]} of project {self.env["GPULAB_PROJECT_NAME"]}."'], shell=True, env=self.env, capture_output=True)
        subprocess.run(['git push origin gpulab'], shell=True, env=self.env, capture_output=True)
        print("GITHUB updated")
        
    def current_commit(self):
        r = subprocess.run(['git rev-parse --short HEAD'], shell=True, env=self.env, capture_output=True)
        return r.stdout.decode('utf-8')[:-1]
        
    def run(self):
        if not GithubAutoUpdater.github_is_known_host():
            GithubAutoUpdater.add_github_to_known_hosts()
        
        if self.github_needs_updating():
            self.update_github()