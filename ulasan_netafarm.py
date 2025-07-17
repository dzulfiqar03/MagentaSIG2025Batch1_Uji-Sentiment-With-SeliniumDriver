from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
from selenium.webdriver.common.by import By
# from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException

def scrape_netafarm():
    print("🚀 Memulai browser...")

    url ="https://www.tokopedia.com/netafarm/review"
    # Tunggu halaman dimuat
    print("⏳ Menunggu halaman dimuat...")
    time.sleep(5)


    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)

    data =[]
    for i in range(0, 10):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        containers = soup.findAll('article', attrs = {'class' : 'css-1pr2lii'})
        for container in containers:
            try:
                nama = container.find('span', attrs = {'data-testid': 'lblItemUlasan'}).text
                produk = container.find('p', attrs = {'class': 'css-akhxpb-unf-heading e1qvo2ff8'}).text
                company = "Netafarm"
                data.append([produk,nama, company])
            except AttributeError:
                continue
        time.sleep(2)
        driver.find_element(By.CSS_SELECTOR,"button[aria-label^='Laman berikutnya']").click()
        time.sleep(3)
    
    # while True:
    #     soup = BeautifulSoup(driver.page_source, "html.parser")
    #     containers = soup.findAll('article', attrs = {'class' : 'css-1pr2lii'})

    #     for container in containers:
    #         try:
    #             nama = container.find('span', attrs = {'data-testid': 'lblItemUlasan'}).text
    #             produk = container.find('p', attrs = {'class': 'css-akhxpb-unf-heading e1qvo2ff8'}).text
    #             company = "Netafarm"
    #             data.append([produk, nama, company])
    #         except AttributeError:
    #             continue

    #     try:
    #         # Cek apakah tombol "Laman berikutnya" masih bisa diklik
    #         next_button = driver.find_element(By.CSS_SELECTOR, "button[aria-label^='Laman berikutnya']")
    #         if next_button.is_enabled():
    #             next_button.click()
    #             time.sleep(3)
    #         else:
    #             print("🚫 Tombol next tidak aktif. Selesai.")
    #             break
    #     except (NoSuchElementException, ElementNotInteractableException):
    #         print("🚫 Tombol next tidak ditemukan. Selesai.")
    #         break
        
    # Simpan ke Excel
    print(data)
   
    print(f"\n✅ Data berhasil disimpan")
    return data



