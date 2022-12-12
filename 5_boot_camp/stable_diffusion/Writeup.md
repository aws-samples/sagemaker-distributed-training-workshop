# Stable Diffusion

### The challenge

You’ve just joined a media and entertainment startup - Octank Digital Media. They are on a mission to disrupt the way humans generate images, the largest innovation in this space since the photograph itself! Your goal - train your own Stable Diffusion model and achieve the highest iterations per second you can, while ensuring your model stays accurate. 

### The dataset

You’ll start with a 100GB sample of the [Laion-5B dataset](https://laion.ai/blog/laion-5b/). This will get you a few million image/text pairs, each downloaded from the internet. 

### The starter code

1/ First, use *this code to download the dataset*. Make sure you point the download script directly to S3, bypassing local disk. 

2/ Second, use *this code to train the model on SageMaker.*

3/ Finally, use *this code to host the model on SageMaker.*

### Milestones and learning objectives

Day 1. Data - Download the data, create an FSx for Lustre volume, and test your script on SageMaker Training.  Your learning objective for the day is creating and managing FSx for Lustre with SageMaker Training.

Day 2. Training - Run on as many GPUs as you can, using data parallel within SageMaker training. Report your best iterations per second per batch size. Ideally run a test with hyperparameter tuning at a small scale to empirically balance cost with accuracy. We do not expect you to train the model to convergence. Your learning objective for the day is to get experience training on multiple GPUs and instances at scale with SageMaker.

Day 3. Hosting - Deploy the model onto a SageMaker Real-time endpoint. Learn about best practices for prompt engineering, such as with [this OpenAI resource](https://cdn.openart.ai/assets/Stable%20Diffusion%20Prompt%20Book%20From%20OpenArt%2010-28.pdf). Spend time learning about what types of prompts work best, and generate the best images you can. Note, you’ll have the option to deploy both the generic pretrained model and the one you’ve just worked with. Your learning objective for the day is to 

### Business pitch and total project cost estimation

Why should senior leadership at Octank Digital Media care about this? Spend some time creating a lightweight business model to see a return from the investment in compute. Using your experience, try to answer the following questions:

1. Where and how could Octank Digital Media deploy a Stable Diffusion model to decrease costs, increase the speed of production, and/or grow their business?
2. Is it worth it for them to train a custom version of this model, or should they simply use the open-source pretrained option? 
3. Stretch question - based on the total expected revenue this model could bring in, what total compute budget could you set? For example, if this were applied in a digital production studio, and increased the yearly return on assets delivered to customers from $150M to $200M, could you easily justify a compute budget of $5M? 

