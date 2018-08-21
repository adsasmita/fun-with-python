p install requests-html
from requests_html import HTMLSession
import re
######## BEGIN INPUTS
url1 = "https://www.kaskus.co.id/thread/5a9f080e9e74041d088b4569/surabaya-jancok-nightlife-surabaya-nightlife---part-5/" #330
url2 = "https://www.kaskus.co.id/thread/5a2d26dcded770232c8b4568/surabaya-jancok-nightlife-surabaya-nightlife---part-4/" #500
url3 = "https://www.kaskus.co.id/thread/59d1fa37de2cf2641f8b4568/surabaya-jancok-nightlife-surabaya-nightlife---part-3/" #500
url4 = "https://www.kaskus.co.id/thread/59028e46ddd77006078b4568/surabaya-jancok-nightlife-surabaya-nightlife---part-2/" #500
url5 = "https://www.kaskus.co.id/thread/58340e0f582b2e39398b456d/surabaya-jancok-nightlife-surabaya-nightlife---part-1/" #494
threads = [url1, url2, url3, url4, url5]
thrlens = [330, 500, 500, 500, 494]
name_to_save = "kaskus_jcsby.txt"
######## END INPUTS
i = 1
txt_to_save = ""
for thread, thrlen in zip(threads, thrlens):
    print(f"Thread: {thread}, Thread Length: {thrlen}")
    links = [f"{thread}/{a}" for a in range(1, thrlen+1)]
    for link in links:
        session = HTMLSession()
        r = session.get(link)
        posts = r.html.find('.entry', clean=True)
        for post in posts:
            txt_to_save += post.text
        print(f"Mined URLs: {i}, Current Text Length: {len(txt_to_save)}")
        i += 1
    print()
txt_to_save = re.compile(pattern).sub('\n', txt_to_save)
to_save = open(name_to_save, "w", encoding='ascii', errors='ignore')
to_save.write(txt_to_save)
to_save.close()


