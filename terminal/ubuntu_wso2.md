STARTING
========

-- To start WSO2 (with logs):
   $ sh api-manager.sh
   To stop:
   $ Ctrl C
-- To start and to stops WSO2 in the background:
   $ sh api-manager.sh start
   $ sh api-manager.sh stop

-- To check that the services are actually running:
   $ ps aux | grep wso2
   Or:
   $ tail -n 20 ~/wso2/wso2am-4.6.0/repository/logs/wso2carbon.log  (logs)
-- To look for error in the server log:
   $ grep -i error ../repository/logs/wso2carbon.log

-- To check that the ports are open:
   $ sudo netstat -tulpn | grep java
   Or:
   $ ss -tulpn | grep 9443
   Or:
   $ curl -k https://localhost:9443/  (HTML response)

   The important ports are:
   9443Admin console (HTTPS)
   8243 API Gateway (HTTPS)
   8280 API Gateway (HTTP)

-- To connect from a Mac Terminal application:
   $ ssh pedro@<ubuntu_vm_ip>
   To get <ubuntu_vm_ip>:
   $ ip a | grep inet  (inside Ubuntu terminal)

-- To suspend a process temporarily (without stoping it) and to put it in the background:
   $ CTRL Z
   $ bg

-- To see the jobs running:
   $ jobs

-- To bring the process back to the foreground:
   $ fg %1 (or whichever job number it shows)

PORTALS
=======

-- To get <VM-IP>:
   $ ip a
   Or:
   $ ip a | grep inet

Management Console (Carbon):
https://<VM-IP>:9443/carbon

API Publisher:
https://<VM-IP>:9443/publisher

Developer Portal:
https://<VM-IP>:9443/devportal

Login (in all of them):
username: admin
password: admin

CONFIGURATION
=============

-- <PRODUCT_HOME>/repository/conf/deployment.toml  --> the main configuration file.
-- <PRODUCT_HOME>/repository/conf/log4j2.properties --> thelog4j2configurationfile

## **Configuring Port Offset in WSO2**

When you run multiple WSO2 products, multiple instances of the same product, or several WSO2 product clusters on the same server or virtual machine (VM),
you must change their default ports using a **port offset** to avoid port conflicts.

The default HTTP and HTTPS ports of a WSO2 product (without any offset) are:

*   **HTTP:** 9763
*   **HTTPS:** 9443

A **port offset** defines the number added to **all** ports defined in the runtime.  
For example:

*   Default HTTP port: **9763**
*   Port offset: **1**
*   Effective HTTP port: **9764**

For each additional WSO2 product instance, you must set a **unique** port offset.  
The default offset is **0**.


## **How to set the port offset**

There are **two ways** to configure a port offset:

***

### **1. Pass the port offset at server startup**

Use the following command to start the server with the default ports incremented by 3:

```sh
./api-manager.sh -DportOffset=3
```

***

### **2. Set the offset in the deployment configuration file**

Edit the file:

    <PRODUCT_HOME>/repository/conf/deployment.toml

Add or modify the following section:

```toml
[server]
offset = 3
```

***











