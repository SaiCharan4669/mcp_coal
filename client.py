import cv2
import base64
import asyncio
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

# image path
IMAGE_PATH = r'C:\Users\ejjig\OneDrive\Desktop\coal_mcp\test.jpg'

# loading and encoding image
def encode_image(path:str)-> str:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'could not load image:{path}')
    _, buffer = cv2.imencode('.jpeg',image)
    return base64.b64encode(buffer).decode('utf-8')


async def main():
    transport = PythonStdioTransport(
        script_path = r'C:\Users\ejjig\OneDrive\Desktop\coal_mcp\server.py'
    )
    
    async with Client(transport)as client:
        image_base64 = encode_image(IMAGE_PATH)
        result = await client.call_tool(
   
                'grade_coal',
                {'image_base64':image_base64}
        )
            
        print("Result:", result.content[0].text)
           

if __name__ == "__main__":
    asyncio.run(main())

    
   