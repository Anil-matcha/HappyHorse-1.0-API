import os
import json
from mcp.server.fastmcp import FastMCP
from happyhorse_api import HappyHorseAPI

# Initialize FastMCP server
mcp = FastMCP("HappyHorse 1.0 API Server")

# Helper to get API client
def get_api():
    return HappyHorseAPI()

@mcp.tool()
def text_to_video(prompt: str, aspect_ratio: str = "16:9", duration: int = 10, quality: str = "1080p", with_audio: bool = False) -> str:
    """
    Generate a native 1080p HD video from a text prompt using HappyHorse 1.0.

    HappyHorse 1.0 (#1 on Artificial Analysis) is Alibaba's 15B-parameter
    Transformer model that jointly generates video and audio in one pass.

    :param prompt: Descriptive text prompt.
    :param aspect_ratio: Video aspect ratio (e.g., '16:9', '9:16', '1:1').
    :param duration: Duration in seconds (default 10).
    :param quality: '1080p' or '4k'.
    :param with_audio: Set True to jointly generate audio alongside the video.
    """
    api = get_api()
    result = api.text_to_video(prompt, aspect_ratio, duration, quality, with_audio)
    return json.dumps(result, indent=2)

@mcp.tool()
def image_to_video(prompt: str, images_list: list[str], aspect_ratio: str = "16:9", duration: int = 10, quality: str = "1080p", with_audio: bool = False) -> str:
    """
    Animate static images into a video using HappyHorse 1.0.

    Reference images in the prompt using @image1, @image2, etc.

    :param prompt: Text prompt guiding the animation.
    :param images_list: List of image URLs to animate.
    :param aspect_ratio: Video aspect ratio.
    :param duration: Duration in seconds.
    :param quality: '1080p' or '4k'.
    :param with_audio: Set True to jointly generate audio alongside the video.
    """
    api = get_api()
    result = api.image_to_video(prompt, images_list, aspect_ratio, duration, quality, with_audio)
    return json.dumps(result, indent=2)

@mcp.tool()
def text_to_video_with_audio(prompt: str, aspect_ratio: str = "16:9", duration: int = 10, quality: str = "1080p") -> str:
    """
    Generate a video with integrated audio from text using HappyHorse 1.0.

    HappyHorse 1.0 generates video and audio jointly in a single Transformer
    forward pass — no separate audio pipeline. Include ambient sound cues in
    the prompt (e.g. 'waves crashing', 'crowd cheering') for richer output.

    :param prompt: Text prompt. Include audio cues for richer sound.
    :param aspect_ratio: Video aspect ratio.
    :param duration: Duration in seconds.
    :param quality: '1080p' or '4k'.
    """
    api = get_api()
    result = api.text_to_video_with_audio(prompt, aspect_ratio, duration, quality)
    return json.dumps(result, indent=2)

@mcp.tool()
def image_to_video_with_audio(prompt: str, images_list: list[str], aspect_ratio: str = "16:9", duration: int = 10, quality: str = "1080p") -> str:
    """
    Animate images into a video with integrated audio using HappyHorse 1.0.

    :param prompt: Text prompt. Reference images with @image1, @image2, etc.
                   Include ambient sound cues for richer audio.
    :param images_list: List of image URLs to animate.
    :param aspect_ratio: Video aspect ratio.
    :param duration: Duration in seconds.
    :param quality: '1080p' or '4k'.
    """
    api = get_api()
    result = api.image_to_video_with_audio(prompt, images_list, aspect_ratio, duration, quality)
    return json.dumps(result, indent=2)

@mcp.tool()
def extend_video(request_id: str, prompt: str = "", duration: int = 10, quality: str = "1080p") -> str:
    """
    Extend a previously generated HappyHorse 1.0 video.

    :param request_id: ID of the video segment to extend.
    :param prompt: Optional prompt to guide the continuation.
    :param duration: Seconds to extend by.
    :param quality: '1080p' or '4k'.
    """
    api = get_api()
    result = api.extend_video(request_id, prompt, duration, quality)
    return json.dumps(result, indent=2)

@mcp.tool()
def video_edit(prompt: str, video_urls: list[str], images_list: list[str] = None,
               aspect_ratio: str = "16:9", quality: str = "1080p") -> str:
    """
    Edit an existing video using natural language with HappyHorse 1.0.

    :param prompt: Describe the desired edits.
    :param video_urls: List of video URLs to edit.
    :param images_list: Optional reference image URLs.
    :param aspect_ratio: Output video aspect ratio.
    :param quality: '1080p' or '4k'.
    """
    api = get_api()
    result = api.video_edit(prompt, video_urls, images_list, aspect_ratio, quality)
    return json.dumps(result, indent=2)

@mcp.tool()
def upload_file(file_path: str) -> str:
    """
    Upload a local file (image or video) to MuAPI for use in generation tasks.

    :param file_path: Local path to the file.
    """
    api = get_api()
    result = api.upload_file(file_path)
    return json.dumps(result, indent=2)

@mcp.tool()
def get_task_status(request_id: str) -> str:
    """
    Check the status and get results of a HappyHorse generation task.

    :param request_id: The ID returned from a generation tool call.
    """
    api = get_api()
    result = api.get_result(request_id)
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    mcp.run()
