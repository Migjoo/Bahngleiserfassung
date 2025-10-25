from pathlib import Path
text = Path('local_models.py').read_text(encoding='utf-8')
start = text.index('for idx, box_info in enumerate(boxes, start=1):')
end = text.index('return annotated', start)
print(text[start:end])
