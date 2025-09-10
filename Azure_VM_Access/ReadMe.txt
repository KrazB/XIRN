PROJECT NAME: Indira

AZURE CONFIGURATION:
USER: kenb-admin@innovationbubble.com.au
dB Password NEW:  Inn0v_tion
dB Password OLD:  Inn0v@tion
dB PasswordORIGINAL:  Lodo104564

AZURE CONFIG MANAGEMENT:
VM Server:   https://portal.azure.com/#@innovationbubble.com.au/resource/subscriptions/d7478fe4-bbba-40cc-ac37-0b69b914176c/resourceGroups/inn-aue-dev-dpp-rg01/providers/Microsoft.Compute/virtualMachines/innaaedppvm01/overview


DETAILS: https://portal.azure.com/#@innovationbubble.com.au/resource/subscriptions/d7478fe4-bbba-40cc-ac37-0b69b914176c/resourceGroups/inn-aue-dev-dpp-rg01/providers/Microsoft.Compute/virtualMachines/innaaedppvm01/connect

PORTS: open to the internet on ports 22, 80 and 443 and 5432 (for postgres server)

VM name: innaaedppvm01

VM Public IP address:4.197.251.181

VM username: azvmadm_inbbl
VM password: 5v&3>gpt8VVl6ed%Ye

################################
activate the role using PIM.
################################
 
If you haven’t done this yet, go here My roles - Microsoft Azure
Click activate on the Contributor role and the Owner and refresh
https://portal.azure.com/#view/Microsoft_Azure_PIMCommon/ActivationMenuBlade/~/azurerbac

Then Start the server if not running

########################
updated Postgres Database Password
########################

PS C:\Users\kenbu> ssh azvmadm_inbbl@4.197.251.181
azvmadm_inbbl@4.197.251.181's password: 5v&3>gpt8VVl6ed%Ye

Welcome to Ubuntu 24.04.1 LTS (GNU/Linux 6.8.0-1021-azure x86_64)

 System information as of Sun Mar 30 22:33:33 UTC 2025

  System load:  0.0                Processes:             249
  Usage of /:   16.6% of 60.95GB   Users logged in:       0
  Memory usage: 0%                 IPv4 address for eth0: 10.20.0.4
  Swap usage:   0%

98 updates can be applied immediately.
To see these additional updates run: apt list --upgradable

13 additional security updates can be applied with ESM Apps.
Learn more about enabling ESM Apps service at https://ubuntu.com/esm

*** System restart required ***
Last login: Thu Feb 27 06:56:48 2025 from 206.83.99.25
azvmadm_inbbl@innaaedppvm01:~$ sudo -i -u postgres
postgres@innaaedppvm01:~$ psql
psql (16.8 (Ubuntu 16.8-0ubuntu0.24.04.1))
Type "help" for help.

postgres=# ALTER USER postgres WITH PASSWORD 'Inn0v_tion';



################################
1.  ACCESS THE VM SERVER VIA PROMPT
################################
1.1 Access the VM (SSH Login) Since the VM is running Ubuntu, access it via SSH: at the COMMAND PROMPT

VM LOGON:
VM username:        	ssh azvmadm_inbbl@4.197.251.181

VM password:     	5v&3>gpt8VVl6ed%Ye


1.2 Elevate Privileges (Get to sudo) Once logged in, gain root privileges:

sudo -i
or
sudo su


####################################
Open Remote Desktop Connection (RDP)
####################################

Press Win + R, type mstsc, and press Enter.
Enter the server’s public IP:

4.197.251.181
Log in with Your Ubuntu Credentials

Use your Ubuntu username and password.
Start GNOME Desktop

If prompted, select GNOME as the session.



####################################
Connect to Postgres DATABASE if using DBeaver
####################################

connecting in dBeaver using:

Host: localhost (or Private IP 10.20.0.4)
Port: 5432
Database: postgres
Username: postgres
Password: (Inn0v_tion)

Windows PC Connection to Remote Postgres Server
root@innaaedppvm01:~# curl ifconfig.me
4.197.251.181root@innaaedppvm01:~#

Open dBeaver on your Windows PC.
Click "Database" → "New Database Connection".
Select PostgreSQL and click Next.

Enter the details:
Host: 4.197.251.181 (your server’s public IP)
Port: 5432
Database: postgres
Username: postgres

Click "Test Connection" → If successful, click "Finish".

To CHANGE PORTS on AZURE if NOT AVAILABLE:  
https://portal.azure.com/#view/Microsoft_Azure_PIMCommon/ActivationMenuBlade/~/azurerbac
Activate My roles | Azure resources


###################################################
#########  PROJECT OUTLINE:  ######################
###################################################

Project Indira: A Data Extraction, Augmentation, and Visualization Framework

Overview:
Project Indira aims to deploy a Microsoft Azure-hosted Ubuntu server to develop and test a dynamic algorithm for processing construction-related data. The primary focus is on reading Industry Foundation Classes (IFC) schema files, as well as other file types stored on a file server.

Key Components:

Data Extraction and Processing:

Algorithm Development: A dynamic algorithm, implemented in Python, will be created to parse IFC schema files and other file formats.
Data Augmentation: Extracted data will be processed and enriched using a series of Pandas DataFrames, transforming raw information into structured, meaningful tables.
Database Integration:

The augmented data is then written to a PostgreSQL database, establishing a robust backend to store and manage the processed information.
Web Service and Visualization:

Backend API: Python web frameworks such as Flask or FastAPI will be employed to create an API that reads from the PostgreSQL database.
Frontend Interface: The API will serve data to a JavaScript-based frontend, which will feature:
Interactive Grid Tables: Utilizing libraries like AG-Grid (or similar) to create dynamic, filterable data tables.
Data Visualizations: Integration with Plotly to generate interactive charts and Three.js to build 3D visualizations.
Purpose and Applications:
The ultimate goal of Project Indira is to validate and ensure the quality of construction data before it is fed into downstream applications. These applications will use the curated datasets to:

Generate registers of quantities.
Develop work breakdown structures.
Assign resources effectively.
Package data for tools involved in planning, costing, and optimizing construction methodologies.


###################################################
#########  PROJECT PLAN:  #########################
###################################################


Okay, given the outline of this project in the instructions,  I have just logged on to a new Microsoft Azure VM Server, here are the details:

USER: kenb-admin@innovationbubble.com.au
PWD: X1234



VM Server:   https://portal.azure.com/#@innovationbubble.com.au/resource/subscriptions/d7478fe4-bbba-40cc-ac37-0b69b914176c/resourceGroups/inn-aue-dev-dpp-rg01/providers/Microsoft.Compute/virtualMachines/innaaedppvm01/overview


DETAILS: https://portal.azure.com/#@innovationbubble.com.au/resource/subscriptions/d7478fe4-bbba-40cc-ac37-0b69b914176c/resourceGroups/inn-aue-dev-dpp-rg01/providers/Microsoft.Compute/virtualMachines/innaaedppvm01/connect

PORTS: open to the internet on ports 22, 80 and 443 

VM name: innaaedppvm01

VM Public IP address:4.197.251.181

VM username: azvmadm_inbbl
VM password: XXX4567 

Let's create a plan to access the Ubuntu that has been setup, confirm the working environment, (ports, services available, capacity, and network connections and speed / latency), tell me how to get to the command prompt (sudo?) and work from there.

After this, let's make a plan to install some basic things like a File Manger (Nemo?), Chrome, dBeaver, Firefox, Thunderbird, Visual Studio, Blender.   Also PostgreSQL, Nginx, Webmin, SSH, Samba, xRDP (any and all if not already provided by Azure).

The emphasis on open source and free applications that are connected and work together.



###################################################
#########  NOTES ON SETUP:  #######################
###################################################

FROM: Michael Burnett
Hi Ken, the bubble change has been approved. 
I've created your account in preparation of the resources being provisioned. 
 
kenb-admin@innovationbubble.com.au

Lodo104564
 
If you open a private browser session and sign into https://portal.azure.com with those credentials it will force you to reset the password and set up MFA.
Microsoft Azure
 
Thanks, as it would have it I am totally under the pump, so will get to this asap
 
No problem. 
I also have a question about the server which I'm just about to hit send on an email about. 
Have a read and just respond whenever you can. Thanks. 
 
Hi Ken, one more question about the VM.
The default option to connect and log in to the VM is with an SSH key pair and it's more secure than a user/password.
 
Just want to make sure you're OK with the key option or if you need it set to user/pw?
 
If I'm understanding the SSH key pair is what I think it is, I found that a right pia in the past.  Given the status and low spec we are at, just the user/pw would be easiest for now, we can upgrade that later, if we need to share it around more than 3-5 people.  Is that okay?   Will users +/- be configured by you, or does that come with a console for me to adjust?
 
Server has been built. You should have just received an email link that has IP, user and password details
 
Link to the VM within Azure is innaaedppvm01 - Microsoft Azure
You can power it on and off from here if needed.
Microsoft Azure
 
There are some specific connection details here as well 
innaaedppvm01 - Microsoft Azure
Microsoft Azure
 
Completely open to the internet on ports 22, 80 and 443 at the moment
 