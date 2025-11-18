"""Main CLI entrypoint for the retail analytics copilot."""

import json
import click
from pathlib import Path
from agent.graph_hybrid import HybridAgent


@click.command()
@click.option('--batch', required=True, help='Path to input JSONL file')
@click.option('--out', required=True, help='Path to output JSONL file')
def main(batch: str, out: str):
    """Run the retail analytics copilot on a batch of questions."""
    
    # Initialize agent
    try:
        agent = HybridAgent()
    except Exception as e:
        click.echo(f"Error initializing agent: {e}", err=True)
        click.echo("Please ensure Ollama is running and the model is pulled:", err=True)
        click.echo("  ollama pull phi3.5:3.8b-mini-instruct-q4_K_M", err=True)
        return
    
    # Read input file
    input_path = Path(batch)
    if not input_path.exists():
        click.echo(f"Error: Input file not found: {batch}", err=True)
        return
    
    questions = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    
    click.echo(f"Processing {len(questions)} questions...")
    
    # Process each question
    results = []
    for i, q in enumerate(questions, 1):
        click.echo(f"Processing question {i}/{len(questions)}: {q['id']}")
        
        try:
            result = agent.process(
                question=q['question'],
                format_hint=q.get('format_hint', 'str')
            )
            
            output = {
                "id": q['id'],
                "final_answer": result['final_answer'],
                "sql": result.get('sql', ''),
                "confidence": result.get('confidence', 0.0),
                "explanation": result.get('explanation', ''),
                "citations": result.get('citations', [])
            }
            
            results.append(output)
            
        except Exception as e:
            click.echo(f"Error processing {q['id']}: {e}", err=True)
            results.append({
                "id": q['id'],
                "final_answer": None,
                "sql": "",
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}",
                "citations": []
            })
    
    # Write output file
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    click.echo(f"Results written to {out}")


if __name__ == '__main__':
    main()

