# Configure FSx for Lustre with SageMaker distributed training
In this guide, you'll follow all the steps to create and configure an FSx for Lustre volume to use with SageMaker distributed training.

### Step 1. Upload data to Amazon S3
The first thing you need is simply an archive of data stored in S3. For me, I'll be pointing to my SageMaker session bucket in us-east-1, using everying in the `data` path.

### Step 2. Create an FSx for Lustre volume
Navigate to FSx in the AWS console. Select "Amazon FSx for Lustre." Optionally give your file system a name, like the name of your dataset. Chose your deployment, throughput, and storage options. If you aren't sure how much data you have in GB's, go back to S3 and use the UI to calculate the total storage of that path.

FSx for Lustre needs to be created in a specific VPC, including the subnet and security group. For the boot camp you can simply use your default VPC, otherwise select your preferred networking settings.

Click on Data Repositry, and select the import from and export to S3. Define these settings, including the local path to be created on Lustre, and the source S3 path. For me, this is `s3://sagemaker-us-east-1-<account id>/data/`.

Create! Lustre should be online in a few minutes.

### Step 3. Configure your VPC to work nicely with SageMaker distributed training
Once Lustre is created, we need to ensure that it can connect to you SageMaker training jobs. You can read more about this in [our docs here](https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html). It needs a few things:

1. Ensure your selected VPC has an internet gateway (IG) attached. If you used the default VPC, then it already has an IG attached. Personally I like to keep a small text file open on my computer where I copy/paste all of my VPC details. 

2. Ensure the subnet where you have launched Lustre has a route to the IG. Go to the VPC page in the AWS console. Click on "route tables." Find your VPC, check the box next to it. Look at "explicit subnet associations." Add an explicit associate from the route table to the subnet where you created FSx for Lustre.

3. Go to security groups in the VPC AWS console page. Check the box next to your VPC. Click on actions, then view details. You need 2 rules for both the inbound and the outbound rules. For the inbound rules, you need one allowing all traffic from your own security group. You also need one allowing all traffic from 0.0.0.0/0. For the outbound rules, you need the same: one allowing all traffic to go to 0.0.0.0/0, and another one allowing all traffic to go to your security group.

4. Go to the Network ACL page in the AWS VPC console. Select the ACL associated with your VPC. Make sure you have explicit allows for both the inbound and outbound rules. I set rule number 1, all traffic allowed from source 0.0.0.0/0. I also set the same for the outbound roles, going to a target destinatino of 0.0.0.0/0.

### Step 4. Create an S3 VPC endpoint
Next, we need to ensure that S3 can receive output files from your VPC. Let's do that now. In the AWS VPC page, select "Endpoints." Click on "create an endpoint." Select "AWS services," then use the search bar to filter down and select 'com.amazonaws.region-s3.' Mine is 'com.amazonaws.us-east-1.s3.' Then select your VPC, along with your route table. I set the policy to "Full access." Click "create," and you should be ready!
    
### Step 5. Test your connection to Lustre from SageMaker training
Finally, we need to make sure that SageMaker training can successfully point to your Lustre volume. To do that, use **[this notebook.](https://github.com/aws-samples/sagemaker-distributed-training-workshop/blob/main/5_boot_camp/lustre/test_lustre.ipynb)**
