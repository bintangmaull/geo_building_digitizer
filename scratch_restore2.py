import json
import os

transcript_path = r'C:\Users\Bintang Maulana\.gemini\antigravity-ide\brain\fc06a54b-dc38-45ea-8afa-36d4b800a2ba\.system_generated\logs\transcript.jsonl'
file_path = r'e:\Project SAM\core\sam_processor.py'

print("Checking out from git...")
os.system(f'git checkout origin/main -- "{file_path}"')

print("Reading file content...")
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

print("Applying edits from transcript...")
with open(transcript_path, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            # Stop right before the buggy Convex Hull experiments at 17:10 WIB (10:10Z)
            if data.get('created_at', '') >= '2026-05-23T10:10:00Z':
                break
            if 'tool_calls' in data:
                for call in data['tool_calls']:
                    if call['name'] == 'replace_file_content':
                        args = call.get('args', {})
                        if 'sam_processor.py' in args.get('TargetFile', ''):
                            target = args['TargetContent']
                            replacement = args['ReplacementContent']
                            if target in content:
                                content = content.replace(target, replacement)
                            else:
                                print('Warning: Target not found at', data.get('created_at'))
                    elif call['name'] == 'multi_replace_file_content':
                        args = call.get('args', {})
                        if 'sam_processor.py' in args.get('TargetFile', ''):
                            for chunk in args['ReplacementChunks']:
                                target = chunk['TargetContent']
                                replacement = chunk['ReplacementContent']
                                if target in content:
                                    content = content.replace(target, replacement)
                                else:
                                    print('Warning: Target chunk not found at', data.get('created_at'))
        except Exception as e:
            pass

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Reconstructed sam_processor.py up to 13:00 WIB!')
