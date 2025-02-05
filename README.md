# 2phaseSIM

## Description
**2phaseSIM** is a 1-D multiphase flow simulator developed in Python. The code allows modeling and analyzing the behavior of two-phase flows in pipelines, using numerical methods to solve governing flow equations.

## Features
- Automatically reads simulation parameters from the `input.json` file
- Validates input parameters
- Executes the simulation with multiphase flow models
- Generates results for analysis

## Repository Structure
```
📂 2phaseSIM
├── 2phaseSIM.py          # Main script for running the simulator
├── input.py              # Input file with flow parameters
├── README.md             # Project documentation
```

## How to Use
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/2phaseSIM.git
   ```
2. Navigate to the project directory:
   ```sh
   cd 2phaseSIM
   ```
3. Edit the `input.json` file to define simulation parameters.
4. Run the simulator:
   ```sh
   python 2phaseSIM.py
   ```
5. The results will be displayed in the terminal and can be analyzed.

## Requirements
- Python 3.x
- Standard libraries (no external dependencies required so far)

## Contribution
Contributions are welcome! To contribute:
1. Fork this repository
2. Create a branch for your feature (`git checkout -b my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the repository (`git push origin my-feature`)
5. Open a Pull Request

## License
This project is distributed under the MIT License.

