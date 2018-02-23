""" elastic_utils.py

Helper functions for Elasticsearch

"""
import os, sys, re
import time

from gmutils.utils import argparser, err
from gmutils.objects import Object

try:
    from boto import kinesis
except Exception as e: err([], {'exception':e, 'level':1})


################################################################################
# OBJECTS

class KinesisStream(Object):
    """
    Simplify interactions with AWS Kinesis Streams

    """
    def __init__(self, stream_name, region='us-east-2'):
        self.stream_name         = stream_name
        self.region              = region
        self.kinesis_conn        = kinesis.connect_to_region(region)    # Connect to a Kinesis stream in a specific region
        self.description         = self.kinesis_conn.describe_stream(stream)
        self.shards              = self.get_shards()
        self.first_shard_id      = self.shards[0]["ShardId"]
        self.partition_key       = 0
        

    def get_shards(self, description):
        """
        Parse shard information

        """
        return self.description['StreamDescription']['Shards']


    def get_partition_key(self):
        """
        Gets an increments the partition key

        """
        self.partition_key += 1
        if self.partition_key >= len(self.shards):
            self.partition_key = 0
            
        return self.partiton_key
        

    def put_record(self, record):
        """
        Put a record (<=1MB) on to a Kinesis stream

        """
        kinesis.put_record(self.stream_name, json.dumps(record), self.get_partition_key())


    def put_records(self, records):
        """
        Put records on to a Kinesis stream

        """
        data = []
        for record in records:
            data.append( { 'Data': json.dumps(record), 'PartitionKey': self.get_partition_key() } )
            
        kinesis.put_records(data, self.stream_name)
        

    def records_iterator(self, limit=100):
        """
        Read the latest batch of records from a Kinesis stream

        """
        shard_it = kinesis.get_shard_iterator(self.stream_name, self.first_shard_id, "LATEST")["ShardIterator"]
        while True:
            t0 = time.time()
            out = kinesis.get_records(shard_it, limit=limit)
            shard_it = out["NextShardIterator"]
            yield out
            
            t1 = time.time()
            if t1 - t0 < 0.2:
                time.sleep(0.2 - (t1 - t0))


        
################################################################################
# MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "Helper functions for AWS Kinesis: kinesis_utils.py"})
    parser.add_argument('--describe_stream', help='Describe a specific Kinesis stream', required=False, type=str)
    args = parser.parse_args()   # Get inputs and options

    if args.describe_stream:
        describe_stream(args.describe_stream)

        
        
################################################################################
