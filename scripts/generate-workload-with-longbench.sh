#!/bin/bash

# # List available tasks                                                                                                                                                                          
# python3 generate_longbench_workload.py --list-tasks                                                                                                                                             
                                                                                                                                                                                                
# # Generate 1000 requests with ~8k tokens (default English tasks)                                                                                                                                
# python3 generate_longbench_workload.py -n 1000 --avg-input-tokens 8000                                                                                                                          
                                                                                                                                                                                                
# # Specific tasks only                                                                                                                                                                           
# python3 generate_longbench_workload.py -n 500 --tasks narrativeqa,gov_report,hotpotqa                                                                                                           
                                                                                                                                                                                                
# # Upload to S3                                                                                                                                                                                  
# python3 generate_longbench_workload.py -n 1000 --avg-input-tokens 10000 --upload-s3       