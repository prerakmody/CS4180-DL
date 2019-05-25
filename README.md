# Topic - Efficient Deep Learning - Model compression via pruning for object detection

# Git Process 
1. Make your own branch called "feature/{sub-project-name}"
    - git checkout -b feature/{sub-project-name}
2. Only push to that branch.
    - git push origin feature/{sub-project-name}
	


# Creating Google Cloud Compute Engine:
There are two ways of doing this.
	1. Create an instance from scratch and install the software requirements(eg. jupyter notebook, tensorflow, ...)
	2. Use the already existing instances in the gcp marketplace which comes with already installed stuff (easier)
	
-----------------------------------
# Common stuff neee to be done:
	1. Click on the link in your webmail if you alredy inserted your email in the brightspace link the teacher provided
	2. Open Google Cloud Platform (gcp) and you must and from the menue on top left open compute engine
	3. you must already have a new project(skip to step 6). If not just create a new project and give it a name 
	4. If you are making a new project, select your billing account as the account you have the credit on
	5. You can check the Billin info from the drop down menue on the top-left to see how much credit you got
	6. *Creat a new VM Instance as follows*
	
# Option 1. Creating a new VM Instance
	1. On the compute engine menue in the left click on VM instances 
	2. Creat new instance
	3. Select the name and the properties of the VM you want
	4. If you want to use Nvidia k80 choose US-central as the region 
		4.1. These are my setting: --> 8CPU- 30GB RAM- Nvidia p100- 100GB Drive
	5. *Check the Boxes* Allow full access to cloud APIs
	6. *Check the Boxes* Allow all traffics to be able to use jupyter note book
	7. Click on create and click on the SSH to connect to the server
	8. If you get error *GPUs All Region Exceeded* follow steps 9-12 if not skip to 13
	9. From the drop down click on IAM & Admin and click on Quotas 
	10. From the metric fliter deselect all the select GPU 
	11. From the location select global and there there should be a compute engine API with limit 0
	12. Select it and click on edit quota and change the limit to 1 add a description and submit
	13.*Don not forget to stop the VM after you are done to aboid the extra charge*
	14. follow this link from step4 [Set up Jupyter](https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52)
	
	
# Option 2. In the search bar search for AISE and select the NVIDIA GPU and lunch it with the setting you want
	1. Write down the Password to the notebook
	1. if the there is an error follow steps 9-12 above
	2. Make an static IP using step 4 and 5 in the link above (by default the port is set to 8888 so if you want u can use it)
	3. open the ssh in the gcp VM and by typing nvidia-smi you should see the gpu information
	4. Type jupyter notebook in the terminal and then from your browser in the windows go to:
		<fixed-external ip of the VM>: <port number-8888 by default> --> 34.34.34.34:8888
	5. Clone the code and start playing around

# Using Tensotboard
	1. run the jupyter notebook file in the setting directory in order to open port 6006 on the vm for tensorboardcolab
	2. run the following commands on the VM to open the port and a tensorboardcplab
	3. ![alt text](https://drive.google.com/file/d/1p4K58topHWHbejfoPfew6yuW8Prl9uUb/view?usp=sharing)

# GPU processes 
	1. check with command *nvidia-smi*
	2. kill a process with command *sudo kill -9 <pid> // sudo kill -9 <pid>*

# Other useful links (GCP VM)
[help1] (https://medium.com/@kn.maragatham09/installing-jupyter-notebook-on-google-cloud-11979e40cd10).

[help2] (https://medium.com/@jamsawamsa/running-a-google-cloud-gpu-for-fast-ai-for-free-5f89c707bae6).

[How to make Virtual Env] (https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-with-python-3-on-ubuntu-18-04).