# Instagram Content Package Generator
This notebook utilizes a CrewAI framework to generate content for Instagram posts about a specified topic. It includes agents for research, writing, reviewing, and image generation.
# Functionality
The notebook performs the following steps:

- Setup: Imports necessary libraries and configures the LLM (OpenAI in this case).
- Image Generation Tool: Defines a tool using diffusers and the "segmind/SSD-1B" model to generate images based on text prompts and saves them locally.
- Agent Definition: Defines four agents:
    -- Researcher: Gathers information on the topic.
    >Writer: Creates short-form and long-form Instagram captions.
    >Reviewer: Polishes and finalizes the captions and hashtags.
    >Image Generator: Creates image prompts and uses the image generation tool.
- Task Definition: Defines the tasks for each agent.
- Crew Creation: Assembles the agents and tasks into a sequential crew.
- Execution: Runs the crew with a given topic, generating research, captions, hashtags, image prompts, and images.
# How to Use
- Install Dependencies: Run the pip install commands in the first few cells if you haven't already.
- Set up API Key: Ensure your OpenAI API key is stored in Colab's user data with the key name '4.1' and the OPENAI_API_KEY environment variable is set (or the api_key parameter in the ChatOpenAI initialization is used).
- Define Topic: Modify the topic variable in the "Execute Crew" section to your desired topic.
- Run All Cells: Execute all the code cells in the notebook.
- View Output: The final output will include the polished captions, hashtags, and the file paths of the generated images. The images themselves will be saved locally in the same directory as the notebook.

# Output Structure

The final output of the crew execution will contain the results from each task, including:
  -A summary of the research.
  -The short-form and long-form Instagram captions.
  -The finalized list of hashtags.
  -The image prompts used and the file paths of the generated images.
The generated images are saved locally as .png files with a timestamp in the filename.
