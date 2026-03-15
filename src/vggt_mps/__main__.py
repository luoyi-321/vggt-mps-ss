#!/usr/bin/env python3
"""
VGGT-MPS: Main entry point for 3D reconstruction with sparse attention
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VGGT 3D Reconstruction on Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with test images
  python main.py demo

  # Process specific images
  python main.py reconstruct data/*.jpg

  # Run with sparse attention
  python main.py reconstruct --sparse data/*.jpg

  # Launch web interface
  python main.py web

  # Run tests
  python main.py test

  # Benchmark performance
  python main.py benchmark
        """
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample images")
    demo_parser.add_argument("--images", type=int, default=2, help="Number of images to use (2-8)")
    demo_parser.add_argument("--kitchen", action="store_true", help="Use kitchen dataset")

    # Reconstruct command
    recon_parser = subparsers.add_parser("reconstruct", help="3D reconstruction from images")
    recon_parser.add_argument("images", nargs="+", help="Image files to process")
    recon_parser.add_argument("--sparse", action="store_true", help="Use sparse attention")
    recon_parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    recon_parser.add_argument("--export", choices=["ply", "obj", "glb"], help="Export format")

    # Web interface command
    web_parser = subparsers.add_parser("web", help="Launch web interface")
    web_parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    web_parser.add_argument("--share", action="store_true", help="Create public link")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--suite", choices=["all", "mps", "sparse", "quick"],
                            default="quick", help="Test suite to run")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark performance")
    bench_parser.add_argument("--mode", choices=["basic", "scaling", "consistency", "ablation-k", "ablation-tau", "ablation-mask", "visualize", "compare-methods"], default="basic",
                             help="Benchmark mode: basic, scaling, consistency, ablation-k, ablation-tau, ablation-mask, visualize, or compare-methods")
    bench_parser.add_argument("--images", type=str, default="10",
                             help="Number of images (single int or comma-separated list)")
    bench_parser.add_argument("--compare", type=str, default=None,
                             help="Methods to compare (comma-separated, e.g., dense,sparse). "
                                  "Use as flag for basic mode or with values for consistency mode.")
    bench_parser.add_argument("--methods", type=str, default="dense,sparse",
                             help="Methods to benchmark (comma-separated: dense,sparse)")
    bench_parser.add_argument("--sparse-k", type=str, default="5,10,20",
                             help="K-nearest values for sparse attention (comma-separated)")
    bench_parser.add_argument("--threshold", type=str, default="0.3,0.5,0.7,0.8,0.9",
                             help="Covisibility threshold values for ablation-tau mode (comma-separated)")
    bench_parser.add_argument("--mask-types", type=str, default="covisibility,random,sliding_window",
                             help="Mask types for ablation-mask mode (comma-separated)")
    bench_parser.add_argument("--sparsity", type=str, default="0.56",
                             help="Target sparsity (single value or comma-separated for compare-methods)")
    bench_parser.add_argument("--metrics", type=str, default="depth_l1,pose_rotation,pose_translation,chamfer",
                             help="Metrics to compute for consistency mode (comma-separated)")
    bench_parser.add_argument("--output", type=str, default=None,
                             help="Output JSON file for results")
    bench_parser.add_argument("--output-dir", type=str, default="results/figures",
                             help="Output directory for visualize mode figures")
    bench_parser.add_argument("--image-dir", type=str, default=None,
                             help="Directory containing real images for benchmarking")
    bench_parser.add_argument("--image-size", type=str, default="640x480",
                             help="Target image size as WxH (e.g., 640x480)")
    bench_parser.add_argument("--recursive", action="store_true",
                             help="Recursively search subdirectories for images")

    # Download model command
    download_parser = subparsers.add_parser("download", help="Download VGGT model")
    download_parser.add_argument("--source", choices=["huggingface", "direct"],
                                default="huggingface", help="Download source")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Import only what we need
    try:
        if args.command == "demo":
            from .commands.demo import run_demo
            run_demo(args)

        elif args.command == "reconstruct":
            from .commands.reconstruct import run_reconstruction
            run_reconstruction(args)

        elif args.command == "web":
            from .commands.web_interface import launch_web_interface
            launch_web_interface(args)

        elif args.command == "test":
            from .commands.test_runner import run_tests
            run_tests(args)

        elif args.command == "benchmark":
            from .commands.benchmark import run_benchmark
            run_benchmark(args)

        elif args.command == "download":
            from .commands.download_model import download_model
            download_model(args)

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(130)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing dependencies: pip install -e .")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"💡 For help, run: vggt --help")
        sys.exit(1)


if __name__ == "__main__":
    main()