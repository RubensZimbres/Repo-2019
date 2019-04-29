import boto3

polly_client = boto3.Session(aws_access_key_id='12345',
                             aws_secret_access_key='ABCD12345/ABC1234',
                             region_name='us-east-1').client('polly')

response = polly_client.start_speech_synthesis_task(VoiceId='Joanna',
                OutputS3BucketName='synth-books',
                OutputS3KeyPrefix='SampleText_key',
                OutputFormat='mp3', 
                Text = 'This an Amazon Polly audio.')

taskId = response['SynthesisTask']['TaskId']

task_status = polly_client.get_speech_synthesis_task(TaskId = taskId)

obj=task_status['SynthesisTask']['OutputUri'][57:]

s3 = boto3.client('s3')

s3.download_file(Bucket='synth-books',Key=obj,Filename=obj)

from pygame import mixer

mixer.init()
mixer.music.load(obj)
mixer.music.play()
