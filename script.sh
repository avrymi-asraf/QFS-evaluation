python3 /home/user/studies/QFS-evaluation/main.py \
  --qfs-path /home/user/studies/Query-Focused-Summarization \
  --run-name STH-01-run-all \
  --input-json /home/user/studies/QFS-evaluation/requests/requests-04.json \
  --output-root /home/user/studies/QFS-evaluation/results \
  --max-iterations 15 \
  --concurrency 5 \
  --meta '{"QuestionGeneratorModel": "gemini-2.5-pro","commitMessage": ""}'