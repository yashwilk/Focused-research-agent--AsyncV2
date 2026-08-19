import os

from focused_research_agent import graph


def main():
    """Render the compiled LangGraph workflow to a PNG file."""
    os.makedirs("diagrams", exist_ok=True)

    png_bytes = graph.build_graph().get_graph().draw_mermaid_png()

    out_path = os.path.join("diagrams", "graph.png")
    with open(out_path, "wb") as f:
        f.write(png_bytes)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
