PyInvestment
===

PyInvestment is still very *pre-alpha* and under active development.

#### Project Goals
  * Provide a simple yet extensible framework to perform real time quantitative financial analysis.
  * Provide a backtesting system to allow the user to test their trading strategy in as realistic of conditions as they see fit.
  * Be simple enough for a beginner to pick up and use while being sophisticated enough that professionals will *need* to use it.
  * Allow the user to use whatever data they want.

### Contributing
Being pre-alpha means that we *need* contributors to help make this project a success. 
Don't hesitate to send me an email with questions or even a pull request with a new feature you think would fit into our framework.


<div class="pagebreak"></div>


### Setup (Ubuntu 64-bit)

#### Docker Installation
Add key server for official Docker repo, update package list, verify repo
```
sudo apt-get adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys 58118E89F3A912897C070ADBF76221572C52609D
sudo apt-get update

```
Verify that you are pointing to official Docker repo (optional)
```
apt-cache policy docker-engine
```

Install Docker, check to see if daemon is running
```
sudo apt-get install docker-engine
sudo systemctl status docker
```

#### Docker Build 
##### (Option 1: local)
Use parameters in local `Dockerfile` to create docker instance
```
cd PyInvestment/

# Format: docker build --tag/-t <user-name>/<repository> .
docker build --tag my/repo . 
docker volume create mongodata
docker run -p 27017:27017 -v mongodata:/data/db my/repo
```

##### (Option 2: repo)
```
docker pull ******
```  
  
