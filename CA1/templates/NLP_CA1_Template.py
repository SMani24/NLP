#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center; padding: 20px; font-family: Vazir;">
# <h1 align="center" style="font-size: 28px; color:rgb(64, 244, 202); width: 100%;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>تمرین 1<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1>
# <h2 dir='rtl' style="color:rgb(90, 255, 184); font-size: 20px;">آشنایی با توکنایزرها و N-gram</h2>
# <p align="center" style="color: #666; font-size: 16px;">شهرزاد آذری آزاد - فرشاد حسامی</p>
# <p align="center" style="color: #666; font-size: 16px; margin-bottom: 30px;">shahrzad.azari@ut.ac.ir - farshad.hessami@ut.ac.ir</p>
# 
# <div dir='rtl' style="border: 2px dashed rgb(90, 255, 184); border-radius: 8px; padding: 20px; margin: 20px auto; max-width: 500px; text-align: right;">
# <p dir='rtl' style="color: rgb(64, 244, 202); font-size: 18px; margin-bottom: 15px;">📝 مشخصات دانشجو:</p>
# <p dir='rtl' style="color: #666; margin: 5px;">نام و نام خانوادگی: {{نام_دانشجو}}</p>
# <p dir='rtl' style="color: #666; margin: 5px;">شماره دانشجویی: {{شماره_دانشجویی}}</p>
# <p dir='rtl' style="color: #666; margin: 5px;">تاریخ ارسال: {{تاریخ_ارسال}}</p>
# </div>
# </div>
# 
# <div dir="rtl" style="text-align: justify; padding: 25px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir; max-width: 100%;word-wrap: break-word;">
# <div style="line-height: 2.0; font-size: 17px; color: black; font-family: Vazir;">
# <div style="padding-right:40px">
# در بخش‌های مختلف این تمرین با مفاهیم Tokenization, Regular Expression , N-gram Language Modeling آشنا می‌شوید و آن‌ها را پیاده‌سازی می‌کنید. 
# </div>
# <br>
# <div style="padding-right:100px">
# 📋 <b>ساختار تمرین:</b>
# <li><b>سوال اول - <span dir="ltr">Regular Expression & Min Distance</span> (20)</b></li>
# <ul>
# <li>بخش اول: تشخیص ایمیل‌های قابل قبول با Regex</li>
# <li>بخش دوم: پیاده‌سازی Auto-Correction با Minimum Edit Distance</li>
# </ul>
# <li><b>سوال دوم - <span dir="ltr">Tokenization</span> (25)</b></li>
# <ul>
# <li>بخش اول: Rule-based Tokenizer</li>
# <li>بخش دوم: BPE Tokenizer</li>
# <li>بخش سوم: Wordpiece Tokenizer</li>
# <li>بخش چهارم: Tokenization Visualization</li>
# </ul>
# <li><b>سوال سوم - <span dir="ltr">N-gram Language Modeling</span> (55)</b></li>
# <ul>
# <li>بخش اول: Data cleaning & Tokenization</li>
# <li>بخش دوم: پیاده‌سازی N-gram</li>
# <li>بخش سوم: معیار Perplexity</li>
# <li>بخش چهارم: روش‌های هموارسازی</li>
# <li>بخش پنجم: شبیه‌سازی Temperature با روش‌های هموارسازی</li>
# </ul>
# </div>
# </div>
# <div dir='rtl' style="line-height: 1.8; font-family: Vazir; font-size: 16px; margin-top: 20px; background-color: #e8eaf6; padding: 15px; border-radius: 8px; color:black">
# 💡 <b>نکات مهم:</b>
# <br>
# در متن سوالات، بخش‌هایی که در آن‌ها مجاز به استفاده از کتابخانه‌های آماده هستید ذکر شده است.
# </div>
# </div>

# # <div style="text-align: center; direction: rtl; font-family: Vazir;"><h1 align="center" style="font-size: 24px; padding: 20px;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>سوال اول - <span dir="ltr">Regular Expression & Min Distance</span> (20)<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1></div>
# 

# <p dir="rtl" style="text-align: right; padding:30px; background-color:rgb(12, 12, 12); border-radius: 12px; color: white; font-family: Vazir;">
# در این سوال شما با نمونه‌هایی عملی از استفاده‌ی Regex و همینطور Minimum Distance مواجه می‌شوید. در بخش اول این سوال، شما باید ایمیل‌های قابل‌قبول را تشخیص داده و در بخش دوم سوال شما با استفاده از Minimum Distance یک سیستم Auto-Correction ساده را پیاده‌سازی خواهید کرد.
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">تشخیص ایمیل‌های قابل قبول با Regex</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# فایل emails.txt که در اختیار شما قرار داده‌شده، شامل تعدادی اسم به‌همراه ایمیل ثبت‌شده‌‌شان در یک سامانه می‌باشد.
# <br>
# از شما خواسته‌شده است تا ایمیل‌هایی که معتبر هستند را با استفاده از regex مشخص کنید.
# <br>
# ایمیل از دو بخش تشکیل می‌شود. که با @ از هم جدا می‌شوند. بخش اول (قبل از @) local-part نام دارد و بخش دوم domain.
# <br>
# منظور از ایمیل معتبر این است که موارد زیر در آن‌ها رعایت شده‌باشند:
# <br>
# ۱. دو بخش ایمیل با یک و تنها یک @ از هم جدا شده‌باشند.
# <br>
# ۲. در local-part هم نام و هم نام خانوادگی شخص وجود داشته باشد.
# <br>
# ۳. در local-part تنها حروف انگلیسی، اعداد و کاراکترهای -،_ و . مجاز هستند. همچنین دو نقطه نمی‌توانند پشت هم بیایند.
# <br>
# ۴. در بخش domain یک میزبان داریم و یک پسوند. میزبان و پسوند همیشه با یک نقطه از یکدیگر جدا می‌شوند. (میزبان می‌تواند در خود نقطه داشته باشد، اما دو نقطه‌ی متوالی در domain مجاز نیست.)
# <br>
# ۵. پسوند از حروف انگلیسی تشکیل می‌شود و حداقل دو کاراکتر دارد.
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - لیست ایمیل‌های قابل قبول موجود در فایل emails.txt
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# در این بخش باید با استفاده از Regex یک تابع بنویسید که اسم یک شخص را به‌عنوان ورودی دریافت کند و درصورت وجود ایمیل معتبر برای این اسم در بین ایمیل‌های ثبت‌شده، ایمیل را برگرداند. در غیر این‌صورت، یک پیغام عدم وجود چاپ کند.
# <br>
# توجه: ممکن است ایمیل این اشخاص، تحت نام شخص دیگری ثبت شده باشد. کار شما این است که ایمیل معتبر این افراد را از بین تمام ایمیل‌ها پیدا کنید.
# <br>
# تابع را با ورودی‌های زیر اجرا کنید:
# <span dir="ltr" style="text-align: left; display: block;">
# name1 = Behnam Khatibi
# </span>
# <span dir="ltr" style="text-align: left; display: block;">
# name2 = Mehrdad Ebrahimi
# </span>
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - ایمیل‌های معتبر مربوط به افراد ذکر شده در توضیحات
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">پیاده‌سازی Auto-Correction با Minimum Edit Distance</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# ۱. ابتدا الگوریتم levenshtein_distance را پیاده‌سازی کنید و با استفاده از آن minimum_distance بین کلمات زیر را به‌دست آورید.
# <span dir="ltr" style="text-align: left; display: block;">
# pair1 = "Athletic", "Atlantic"
# </span>
# <span dir="ltr" style="text-align: left; display: block;">
# pair2 = "London", "Boston"
# </span>
# <span dir="ltr" style="text-align: left; display: block;">
# pair3 = "Action", "Compact"
# </span>
# <span dir="ltr" style="text-align: left; display: block;">
# pair3 = "", "Sting"
# </span>
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - مقدار minimum distance بین جفت کلمه‌های داده‌شده در بخش توضیحات
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# ۲. بعد از پیاده‌سازی minimum distance حال باید با استفاده از آن، جمله‌ی زیر را اصلاح املایی کنید. در این جمله تعدادی کلمه وجود دارند که املایشان نادرست است. یک لیست از املای صحیح کلمات که کلمات این جمله را نیز شامل می‌شوند در فایل vocab.txt موجود هستند.
# <span dir="ltr" style="text-align: left; display: block;">
# sentence_to_be_corrected = 
# "helo studnts at the universty are wrting ther frst edit distnce algorthm in pythn, and they reely enjy it!"
# </span>
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - اصلاح‌شده‌ی جمله‌ی داده‌شده در بخش توضیحات با کمک فایل vocab.txt
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# # <div style="text-align: center; direction: rtl; font-family: Vazir;"><h1 align="center" style="font-size: 24px; padding: 20px;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>سوال دوم - <span dir="ltr">Tokenization</span> (25)<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1></div>
# 

# <p dir="rtl" style="text-align: right; padding:30px; background-color:rgb(12, 12, 12); border-radius: 12px; color: white; font-family: Vazir;">
# در این سوال، با انواع مختلف توکنایزر و روش  پیاده‌سازی آن‌ها آشنا می‌شوید.
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">Rule-based Tokenizer</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# با کمک دستورات regex یک توکنایزر بنویسید که متن را با استفاده از علائم نگارشی تقسیم کند.
# <br>
# سپس عملکرد آن را بر روی جملات داده شده زیر آزمایش کنید.
# <span dir="ltr" style="text-align: left; display: block;">
# "Hello, world! NLP is fun."
# </span>
# <span dir="ltr" style="text-align: left; display: block;">
# "That U.S.A. poster-print costs $12.40..."
# </span>
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - تعداد و لیست توکن‌های ایجادشده با توکنایزر صورت سوال برای هر جمله<br>
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# ایرادات این توکنایزر چیست؟
# <br>
# چند راه‌حل برای بهبود عملکرد این توکنایزر ارائه دهید.
# <br>
# یکی از آن‌ها را پیاده‌سازی کنید.
# <br>
# عملکرد توکنایزر جدید را بر روی جملات داده شده آزمایش کنید.
# <span dir="ltr" style="text-align: left; display: block;">
# "Hello, world! NLP is fun."
# </span>
# <span dir="ltr" style="text-align: left; display: block;">
# "That U.S.A. poster-print costs $12.40..."
# </span>
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - تعداد و لیست توکن‌های ایجادشده با توکنایزر بهبودیافته برای هر جمله
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش اول:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">BPE Tokenizer</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# ابتدا دیتاست TinyStories-Farsi را از HuggingFace لود کنید.
# <a href="https://huggingface.co/datasets/taesiri/TinyStories-Farsi" target="_blank" rel="noopener noreferrer" style="color: #9EEAD2; text-decoration: underline;">
#     (لینک دیتاست)
# </a>
# <br>
# سپس از داده آموزش (train) آن جملات فارسی را در یک لیست اضافه کنید.
# <br>
# اکنون با استفاده از این مجموعه داده، یک توکنایزر BPE آموزش دهید. استفاده از کتابخانه‌های آماده مانعی ندارد.
# <br>
# عملکرد توکنایزر را روی جمله زیر آزمایش کنید:
# <br>
# "روزی یک مرد ثروتمند، پسر بچه کوچکش را بـه ده برد تا بـه او نشان دهد مردمی که در آنجا زندگی می‌کنند، چقدر فقیر هستند."
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - مجموعه داده آموزش<br>
# - تعداد و لیست توکن‌های ایجادشده برای جمله
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">Wordpiece Tokenizer</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# درباره Wordpiece Tokenizer تحقیق کنید.
# <br>
# نحوه آموزش این توکنایزر را به طور دقیق شرح دهید و سپس آن را با BPE مقایسه کنید.
# <br>
# اکنون یک توکنایزر Wordpiece بر روی مجموعه داده خود آموزش دهید. استفاده از کتابخانه‌های آماده مانعی ندارد.
# <br>
# عملکرد توکنایزر را روی جمله زیر آزمایش کنید:
# <br>
# "روزی یک مرد ثروتمند، پسر بچه کوچکش را بـه ده برد تا بـه او نشان دهد مردمی که در آنجا زندگی می‌کنند، چقدر فقیر هستند."
# </p>

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - تعداد و لیست توکن‌های ایجادشده برای جمله
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش سوم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">Tokenization Visualization</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# با استفاده از ابزار 
# <a href="https://tiktokenizer.vercel.app/" target="_blank" rel="noopener noreferrer" style="color: #9EEAD2; text-decoration: underline;">
#     tiktokenizer
# </a>
# توکن‌های تولید شده برای جمله زیر را در هر یک از مدل‌های gpt2 و gpt4 و Meta-Llama-3-8B را مشاهده و تفاوت‌ها را گزارش کنید.
# <br>
# "روزی یک مرد ثروتمند، پسر بچه کوچکش را بـه ده برد تا بـه او نشان دهد مردمی که در آنجا زندگی می‌کنند، چقدر فقیر هستند."
# <br>
# سپس در مورد توکنایزر استفاده شده در هر یک تحقیق کنید. به نظر شما علت تفاوت نتیجه آن‌ها در چیست؟
# </p>

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - توکن‌های ایجادشده با هر توکنایزر برای جمله
# </p>

# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش چهارم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# # <div style="text-align: center; direction: rtl; font-family: Vazir;"><h1 align="center" style="font-size: 24px; padding: 20px;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>سوال سوم - <span dir="ltr">N-gram Language Modeling</span> (55)<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1></div>
# 

# <p dir="rtl" style="text-align: right; padding:30px; background-color:rgb(12, 12, 12); border-radius: 12px; color: white; font-family: Vazir;">
# در این سوال، با N-gram Language Modeling و آن را پیاده‌سازی می‌کنید، با استفاده از آن به تولید متن می‌پردازید. سپس با معیار perplexity و نحوه کاربرد آن آشنا می‌شوید. در نهایت الگوریتم‌های smoothing و با کاربردهای آن آشنا می‌شوید. را پیاده‌سازی می‌کنید
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">Data cleaning & Tokenization</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# مجموعه داده 
# <a href="https://huggingface.co/datasets/taesiri/TinyStories-Farsi" target="_blank" rel="noopener noreferrer" style="color: #9EEAD2; text-decoration: underline;">
#     TinyStories-Farsi
# </a>
# - که در سوال اول نیز از آن استفاده کردید - را لود کنید.
# <br>
# ابتدا دادگان فارسی در مجموعه آموزش (train) آن را تمیز کرده و پیش‌پردازش‌های مورد نیاز را بر روی آن انجام دهید. (با بررسی داده‌ها، مشخص‌کنید که این دادگان به چه پیش‌پردازش‌هایی نیاز دارند.)
# <br>
# سپس با استفاده از BPE Tokenizer که بر روی این دادگان آموزش‌داده‌اید، دادگان پردازش‌شده را توکنایز کنید.
# <br>
# توکن‌های یک جمله را به انتخاب خود، چاپ کنید.
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b>
# <br>
# - مجموعه داده تمیزشده فارسی
# <br>
# - مجموعه داده توکنایز شده فارسی
# <br>
# - توکن‌های یک جمله‌ی دلخواه
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش اول:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">پیاده‌سازی N-gram</div>
# 

# <p dir='rtl' style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# یک کلاس برای ساخت و آموزش N-gram بنویسید.
# <br>
# با استفاده از این کلاس و مجموعه داده توکنایز شده که در بخش قبل آماده کردید، 
# <span dir="ltr"> 2-gram, 4-gram, 8-gram</span>
# بسازید و روی مجموعه داده خود آموزش دهید.
# <br>
# با استفاده از مدل‌های آموزش داده شده، متن‌های 100 توکنی تولید کنید و کیفیت متون تولیدشده را با هم مقایسه کنید و تفاوت عملکرد مدل‌ها از نظر پیوستگی و روانی متون را تحلیل کنید.
# </p>

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - مدل‌های N-gram آموزش یافته
# <br>
# - متن‌های 100 توکنی تولیدشده با هر N-gram
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش دوم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">معیار Perplexity</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# ابتدا از مجموعه داده 
# <a href="https://huggingface.co/datasets/taesiri/TinyStories-Farsi" target="_blank" rel="noopener noreferrer" style="color: #9EEAD2; text-decoration: underline;">
#     TinyStories-Farsi
# </a>
# دادگان validation فارسی را جدا کنید.
# <br>
# سپس معیار Perplexity را برای هر کدام از N-gram های خود، روی این مجموعه حساب کنید.
# <br>
# تحلیل خود را از این نتیجه بگویید.
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - مجموعه دادگان فارسی validation
# <br>
# - نتیجه معیار Perplexity در هر N-gram
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش سوم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# <p dir='rtl' style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# در این قسمت چهار جمله در اختیار شما قرار گرفته است. استدلال کنید کدام جمله‌ها محتمل‌ترند که توسط یک <span dir="ltr">4-gram</span> که روی داده‌های مشابه آموزش دیده‌است، تولید شده باشند.
# <br>
# جمله‌ی اول = آنها دوست داشتند در ماسه بازی کنند و جزر و مد آب را تماشا کنند
# <br>
# جمله‌ی دوم = جیل و تام به همراه مامان و بابا به ساحل رفتند
# <br>
# جمله‌ی سوم = تام بطری‌ نوشابه‌ را تا حد ممکن بالا انداخت و به سمت او فریاد زد
# <br>
# جمله‌ی چهارم = باری خیلی دوست داشت بیرون از منزل نقاشی کند و با پدربزرگ منظره تماشا کند
# </p>

# In[ ]:




# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش چهارم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">روش‌های هموارسازی</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# هر کدام از الگوریتم‌های هموارسازی (Smoothing) که در درس خوانده‌اید (Laplace, Interpolation, Backoff) را پیاده‌سازی کنید.
# <br>
# یک مدل
# <span dir="ltr">4-gram</span>
# بسازید و با مجموعه داده توکنایزشده فارسی که در بخش اول این سوال آماده کردید، آموزش دهید.
# (می‌توانید از مدل 
# <span dir="ltr">4-gram</span>
# آموزش‌یافته قبلی خود استفاده کنید.)
# <br>
# در الگوریتم Interpolation مقادیر λ را 0.4, 0.3, 0.2, 0.1 در نظر بگیرید. به این صورت:
# P​(w∣h3​,h2​,h1​)=0.4P​(w∣h3​,h2​,h1​)+0.3P​(w∣h2​,h1​)+0.2P​(w∣h1​)+0.1P​(w)
# <br>
# در الگوریتم Backoff مقدار λ را 0.4 در نظر بگیرید.
# <br>
# سپس با استفاده از هر یک از این روش‌ها، متن‌هایی به طول 100 توکن تولید کنید.
# <br>
# متن‌های تولیدشده را با یکدیگر مقایسه کنید و تفاوت آن‌ها را از نظر روانی و تنوع کلمات بررسی کنید.
# <br>
# در نهایت، با استفاده از احتمالات محاسبه‌شده در هر یک از این روش‌ها، معیار Perplexity را روی مجموعه دادگان فارسی validation - که در بخش سوم این سوال آماده کردید - حساب کنید.
# <br>
# مقادیر Perplexity برای هر کدام از روش‌های هموارسازی و مدل غیرهموار (Unsmoothed) را با یکدیگر مقایسه کنید و تحلیل خود را از این نتایج بگویید.
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - متن‌های تولیدشده با هر یک از روش‌های هموارسازی ذکر شده
# <br>
# - معیار Perplexity برای هر یک از روش‌های هموارسازی ذکر شده
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش پنجم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">پیاده‌سازی Temperature</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# در مورد تاثیر Temperature در مدل‌های زبانی توضیح‌دهید.
# <br>
# به هنگام Sampling از N-gram، برای آن Temperature تعریف کنید و پیاده‌سازی‌های لازم را انجام دهید.
# <br>
# با قرار دادن کمترین Temperature و با بیشترین Temperature، سه‌بار خروجی‌هایی به طول ۲۰ توکن تولید کنید. (برای هر حالت سه‌بار) و سپس تاثیر Temperature در خروجی‌ها را تحلیل کنید.
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - متن‌های تولید‌شده با Temperature بالا و پایین (برای هریک ۳ عدد متن تولید شده)
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی زیربخش ششم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# # <h1 style="text-align: right;">**نکات مهم و قوانین تحویل**</h1>

# 
# <div dir="rtl" style="font-family: Vazir; width: 85%; font-size: 16px; text-align: right;">
#     <p style="text-align: right;" dir="rtl"><strong dir="rtl">مهلت تحویل :</strong> 10 آبان</p>
# </div>
# <h4 dir="rtl" style="font-family: Vazir; width: 85%;">فایل ارسالی شما باید با فرمت زیر نامگذاری شود: <code>NLP_CA{n}_{LASTNAME}_{STUDENTID}.ipynb</code></h4>
# <h4 dir="rtl" style="font-family: Vazir; width: 85%;">نحوه انجام تمرین:</h4>
# <ul dir="rtl" style="font-family: Vazir; width: 85%; font-size: 16px;">
#   <li>سلول‌های کد با برچسب <code>WRITE YOUR CODE HERE</code> را تکمیل کنید.</li>
#   <li>برای پاسخ‌های متنی، متن <code>{{پاسخ_خود_را_اینجا_بنویسید}}</code> را با پاسخ خود جایگزین کنید.</li>
# </ul>
# <h4 dir="rtl" style="font-family: Vazir; width: 85%;">صداقت علمی:</h4> <ul dir="rtl" style="font-family: Vazir; width: 85%; font-size: 16px;"> <li>ما نوت‌بوک‌های تعداد مشخصی از دانشجویان که به صورت تصادفی انتخاب می‌شوند، بررسی خواهیم کرد. این بررسی‌ها اطمینان حاصل می‌کنند که کدی که نوشتید واقعاً پاسخ‌های موجود در نوت‌بوک شما را تولید می‌کند. اگر پاسخ‌های صحیح را در نوت‌بوک خود بدون کدی که واقعاً آن پاسخ‌ها را تولید کند تحویل دهید، این یک مورد جدی از عدم صداقت علمی محسوب می‌شود.</li> <li>ما همچنین بررسی‌های خودکاری را برای تشخیص سرقت علمی در نوت‌بوک‌های کولب انجام خواهیم داد. کپی کردن کد از دیگران نیز یک مورد جدی از عدم صداقت علمی محسوب می‌شود.</li> </ul>
# <h4 dir="rtl" style="font-family: Vazir; width: 85%;">توضیحات تکمیلی:</h4> <ul dir="rtl" style="font-family: Vazir; width: 85%; font-size: 16px;">
# <li>
# خوانایی و دقت بررسی‌ها در گزارش نهایی از اهمیت ویژه‌ای برخوردار است. به تمرین‌هایی که به صورت کاغذی تحویل داده شوند یا به صورت عکس در سایت بارگذاری شوند، ترتیب اثری داده نخواهد شد.</li>
# <li>
#  همه‌ی کدهای پیوست گزارش بایستی قابلیت اجرای مجدد داشته باشند. در صورتی که برای اجرا مجدد آن‌ها نیاز به تنظیمات خاصی می‌باشد، بایستی تنظیمات مورد نیاز را نیز در گزارش خود ذکر کنید.  دقت کنید که  تمامی کدها باید توسط شما اجرا شده باشند و نتایج اجرا در فایل کدهای ارسالی مشخص باشد. به کدهایی که نتایج اجرای آن‌ها در فایل ارسالی مشخص نباشد نمره‌ای تعلق نمی‌گیرد.
# </li>
# <li>توجه کنید این تمرین باید به صورت تک‌نفره انجام شود و پاسخ‌های ارائه شده باید نتیجه فعالیت فرد نویسنده باشد (همفکری و به اتفاق هم نوشتن تمرین نیز ممنوع است). در صورت مشاهده
#  تشابه به همه افراد مشارکت‌کننده، نمره تمرین صفر و به استاد گزارش می‌گردد.
#  </li>
# 
#  <li>
# لطفاً تمامی پاسخ‌های متنی خود را با <b>فونت وزیر (Vazir)</b> و به‌صورت <b>راست‌چین</b> بنویسید.  
# از استفاده از فونت‌های پیش‌فرض خودداری کنید تا ظاهر نوت‌بوک شما یک‌دست و خوانا باشد.  
# در بخش‌های تشریحی، سعی کنید پاسخ‌ها را کامل، منسجم و با رعایت نگارش فارسی بنویسید.  
# همچنین، به چینش تمیز سلول‌ها و اجرای درست کدها توجه کنید تا تمرین شما با فرمت خواسته‌شده و استاندارد ارائه شود.
# </li>
#  <li>برای مطالعه بیشتر درباره‌ی فرمت Markdown می‌توانید از <a href="https://github.com/tajaddini/Persian-Markdown/blob/master/learn-MD.md">این لینک</a> مطالعه کنید.
#  </li>
#  </ul>
#     
# 
#  </div>
# 
