# Sign language understanding through HCI


## Why did we do this?
The project has been put together for the submission of the JacobsHack 2019. 
Nowadays there are multiple smart accessories such as smart glasses, smartwatches, etc. available in the market. But all of them are built for us, people that have no physical impairments. As a matter of fact, our project stems from the latter lack of technology to aid the physically challenged. 

Specifically, our project focuses on aphasic or speech impaired people, our main aim is to construct a bridge between this target of people and new smart assistant that relies completely on voice (Google Dot, Amazon Alexa etc.).

The projects, in a nutshell, uses machine learning pipeline: first of all, it recognises human hands, it then maps them to letters from the American Sign Language and finally, it maps them to real alphabet letters. After having recognised a certain letter or sign it will utter a command trough speakers (text to speech is achieved trough GoogleTTS Cloud API) and order Alexa (in this specific case) to do something accordingly. 

The principle behind it is merely to showcase that the idea behind our project is possible and could be taken to a full implementation level, but for now, it remains a baseline for future improvements.
So far we have been only able to train the model on three letters, but our idea is to complete the alphabet by using a bigger and more detailed dataset.

Also, further improvement sees the project to be more self-contained and portable. The latter would be achieved by running the model on a microcontroller (e.g. RaspberryPi) equipped with a camera and speakers.


