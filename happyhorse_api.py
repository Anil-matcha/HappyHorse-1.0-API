import os
import requests
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class HappyHorseAPI:
    def __init__(self, api_key=None):
        """
        Initialize the HappyHorse 1.0 API client.
        :param api_key: Your MuAPI.ai API key. Defaults to MUAPI_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("MUAPI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Set MUAPI_API_KEY in .env or pass it to the constructor.")

        self.base_url = "https://api.muapi.ai/api/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def text_to_video(self, prompt, aspect_ratio="16:9", duration=10, quality="1080p", with_audio=False):
        """
        Submits a HappyHorse 1.0 Text-to-Video (T2V) generation task.

        HappyHorse 1.0 generates native 1080p HD video with a single 40-layer
        Transformer (15B parameters). Audio can be generated jointly in one pass
        by setting with_audio=True.

        :param prompt: The text prompt describing the video.
        :param aspect_ratio: Video aspect ratio (e.g., '16:9', '9:16', '1:1').
        :param duration: Video duration in seconds (default 10).
        :param quality: Output quality ('1080p' or '4k').
        :param with_audio: Whether to generate audio alongside the video in a single pass.
        :return: JSON response with request_id.
        """
        endpoint = f"{self.base_url}/happyhorse-1.0-t2v"
        if with_audio:
            endpoint = f"{self.base_url}/happyhorse-1.0-t2v-audio"
        payload = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "quality": quality,
        }
        return self._post_request(endpoint, payload)

    def image_to_video(self, prompt, images_list, aspect_ratio="16:9", duration=10, quality="1080p", with_audio=False):
        """
        Submits a HappyHorse 1.0 Image-to-Video (I2V) generation task.

        Animate one or more static images into a video. Reference images in the
        prompt using @image1, @image2, etc.

        :param prompt: Text prompt to guide the animation. Reference images with
                       @image1, @image2, etc.
        :param images_list: A list of image URLs to animate.
        :param aspect_ratio: Video aspect ratio.
        :param duration: Video duration in seconds.
        :param quality: Output quality ('1080p' or '4k').
        :param with_audio: Whether to generate audio alongside the video.
        :return: JSON response with request_id.
        """
        endpoint = f"{self.base_url}/happyhorse-1.0-i2v"
        if with_audio:
            endpoint = f"{self.base_url}/happyhorse-1.0-i2v-audio"
        payload = {
            "prompt": prompt,
            "images_list": images_list,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "quality": quality,
        }
        return self._post_request(endpoint, payload)

    def text_to_video_with_audio(self, prompt, aspect_ratio="16:9", duration=10, quality="1080p"):
        """
        Submits a HappyHorse 1.0 Text-to-Video with integrated audio generation.

        HappyHorse jointly generates video and audio in a single Transformer
        forward pass — no separate audio pipeline needed.

        :param prompt: The text prompt describing the scene (include audio cues,
                       e.g. 'rain pattering', 'crowd cheering' for richer audio).
        :param aspect_ratio: Video aspect ratio.
        :param duration: Video duration in seconds.
        :param quality: Output quality ('1080p' or '4k').
        :return: JSON response with request_id.
        """
        return self.text_to_video(prompt, aspect_ratio, duration, quality, with_audio=True)

    def image_to_video_with_audio(self, prompt, images_list, aspect_ratio="16:9", duration=10, quality="1080p"):
        """
        Submits a HappyHorse 1.0 Image-to-Video with integrated audio generation.

        :param prompt: Text prompt guiding animation. Include ambient sound cues for
                       richer audio. Reference images with @image1, @image2, etc.
        :param images_list: List of image URLs to animate.
        :param aspect_ratio: Video aspect ratio.
        :param duration: Video duration in seconds.
        :param quality: Output quality ('1080p' or '4k').
        :return: JSON response with request_id.
        """
        return self.image_to_video(prompt, images_list, aspect_ratio, duration, quality, with_audio=True)

    def extend_video(self, request_id, prompt="", duration=10, quality="1080p"):
        """
        Extends a previously generated HappyHorse 1.0 video.

        :param request_id: The request_id of the video segment to extend.
        :param prompt: Optional text prompt to guide the continuation.
        :param duration: Seconds to extend by.
        :param quality: Output quality ('1080p' or '4k').
        :return: JSON response with request_id.
        """
        endpoint = f"{self.base_url}/happyhorse-1.0-extend"
        payload = {
            "request_id": request_id,
            "prompt": prompt,
            "duration": duration,
            "quality": quality,
        }
        return self._post_request(endpoint, payload)

    def video_edit(self, prompt, video_urls, images_list=None, aspect_ratio="16:9", quality="1080p"):
        """
        Edits an existing video using natural language with HappyHorse 1.0.

        :param prompt: Describe the desired edits.
        :param video_urls: List of video URLs to edit.
        :param images_list: Optional list of reference image URLs.
        :param aspect_ratio: Output video aspect ratio.
        :param quality: Output quality ('1080p' or '4k').
        :return: JSON response with request_id.
        """
        endpoint = f"{self.base_url}/happyhorse-1.0-video-edit"
        payload = {
            "prompt": prompt,
            "video_urls": video_urls,
            "images_list": images_list or [],
            "aspect_ratio": aspect_ratio,
            "quality": quality,
        }
        return self._post_request(endpoint, payload)

    def _post_request(self, endpoint, payload):
        response = requests.post(endpoint, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def upload_file(self, file_path):
        """
        Uploads a local file (image or video) to MuAPI for use in generation tasks.

        :param file_path: Path to the local file to upload.
        :return: JSON response containing the URL of the uploaded file.
        """
        endpoint = f"{self.base_url}/upload_file"
        headers = {
            "x-api-key": self.api_key
        }
        with open(file_path, "rb") as file_data:
            files = {"file": file_data}
            response = requests.post(endpoint, headers=headers, files=files)
        response.raise_for_status()
        return response.json()

    def get_result(self, request_id):
        """
        Polls for the result of a HappyHorse generation task.

        :param request_id: The request_id returned from a generation call.
        :return: JSON response with status and outputs.
        """
        endpoint = f"{self.base_url}/predictions/{request_id}/result"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def wait_for_completion(self, request_id, poll_interval=5, timeout=600):
        """
        Blocks until a HappyHorse generation task completes and returns the result.

        :param request_id: The request_id returned from a generation call.
        :param poll_interval: Seconds between status polls (default 5).
        :param timeout: Maximum seconds to wait before raising TimeoutError (default 600).
        :return: Completed result JSON with 'outputs' list.
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            result = self.get_result(request_id)
            status = result.get("status")

            if status == "completed":
                return result
            elif status == "failed":
                raise Exception(f"Video generation failed: {result.get('error')}")

            print(f"Status: {status}. Waiting {poll_interval} seconds...")
            time.sleep(poll_interval)

        raise TimeoutError("Timed out waiting for HappyHorse video generation to complete.")


if __name__ == "__main__":
    # Example usage for T2V
    try:
        api = HappyHorseAPI()
        prompt = "A cinematic aerial shot of a coastal city at golden hour, waves crashing, 1080p"

        print(f"Submitting T2V task with prompt: {prompt}")
        submission = api.text_to_video(prompt=prompt, duration=10)
        request_id = submission.get("request_id")
        print(f"Task submitted. Request ID: {request_id}")

        print("Waiting for completion...")
        result = api.wait_for_completion(request_id)
        print(f"Generation completed! Video URL: {result.get('outputs', [None])[0]}")

    except Exception as e:
        print(f"Error: {e}")
