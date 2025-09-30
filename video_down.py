import yt_dlp


def download_youtube_video_no_audio(url, output_path="./videos"):
    ydl_opts = {
        "outtmpl": f"{output_path}/%(title)s.%(ext)s",
        # Sadece video stream (ses yok)
        "format": "bestvideo[ext=mp4]",
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


if __name__ == "__main__":
    video_url = input("YouTube URL'sini gir: ")
    download_youtube_video_no_audio(video_url, "./videos")
