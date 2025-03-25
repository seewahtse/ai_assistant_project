import json

with open('config.json') as config_file:
    config = json.load(config_file)

assembly_ai_key = config['assembly_ai_key']
eleven_labs_key = config['eleven_labs_key']