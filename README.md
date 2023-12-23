This project looks to familiarize myself with image classification networks.
To do so, images taken from 3 Star Wars Battlefront 2 maps were classified to identify the map being played on, and also the region of the map shown in the current image.

An image classification network was created in MATLAB using the trainnetwork function. Multiple layer configurations were tested, and in general this model performed poorly with a testing/validation accuracy of at most ~70%,
but in general performed between 60-70%. 

A prefabricated image classificaion network with deep learning (resnet-50) was then used for the same purpose, and an accurayc of over 80% was achieved.
