# AI 2027 Timelines and Takeoff Simulations

This is the code to run the simulations for AI 2027's [timelines forecast](https://ai-2027.com/research/timelines-forecast) and [takeoff forecast](https://ai-2027.com/research/takeoff-forecast).

Use `poetry install` to install needed packages.

To run the original timelines simulation, input your parameters in the `timelines/params.yaml` file. 

Then, cd into the `timelines` folder and run `poetry run python forecasting_timelines.py` to run the benchmarks and gaps method. Run `simple_forecasting_timelines.py` to run the time horizon extension method. The results are saved in the `figures` folder.

Do the same thing but in the `timelines_may_2025_update` folder to run the May 2025 version of the timelines model.

To run the takeoff simulation, input your parameters in `takeoff/takeoff_params.yaml`, cd into `takeoff` then run `forecasting_takeoff.py`.

Note that all time amounts in the timelines simulation are in "work time", i.e. the amount of time that a human would work during that time period: for example, a work week is 40 hours and a work year is 2,000 hours.

We'd love for others run the simulations with their own parameters or extend our work. For example, due to time constraints we weren't able to incorporate training compute increases or uncertainty over AI R&D progress multipliers in our takeoff simulation.

The simulation was implemented by Nikola Jurkovic and Eli Lifland.