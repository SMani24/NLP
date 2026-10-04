#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center; padding: 20px; font-family: Vazir;">
# <h1 align="center" style="font-size: 28px; color:rgb(64, 244, 202); width: 100%;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>تمرین سوم<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1>
# <h2 style="color:rgb(90, 255, 184); font-size: 20px;">Word2Vec & MLP</h2>
# <p align="center" style="color: #666; font-size: 16px;">علی فرتوت</p>
# <p align="center" style="color: #666; font-size: 16px; margin-bottom: 30px;">ali.fartout@ut.ac.ir</p>
# 
#     
# <p align="center" style="color: #666; font-size: 16px;">علیرضا آخوندی</p>
# <p align="center" style="color: #666; font-size: 16px; margin-bottom: 30px;">a.akhoundi79@gmail.com</p>
# 
# <div dir="rtl" style="border: 2px dashed rgb(90, 255, 184); border-radius: 8px; padding: 20px; margin: 20px auto; max-width: 500px; text-align: right;">
# <p style="color: rgb(64, 244, 202); font-size: 18px; margin-bottom: 15px;">📝 مشخصات دانشجو:</p>
# <p style="color: #666; margin: 5px;">نام و نام خانوادگی: {{نام_دانشجو}}</p>
# <p style="color: #666; margin: 5px;">شماره دانشجویی: {{شماره_دانشجویی}}</p>
# <p style="color: #666; margin: 5px;">تاریخ ارسال: {{تاریخ_ارسال}}</p>
# </div>
# </div>
# 
# <div dir="rtl" style="text-align: justify; padding: 25px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# <div style="line-height: 2.0; font-size: 17px; color: black; font-family: Vazir;">
# 
# <br>
# <div style="padding-right:100px">
# 📋 <b>ساختار تمرین:</b>
# <li><b>سوال اول - <span dir="rtl">سوال اول : پیاده سازی CBOW و Skip-Gram</span> </b></li>
# <ul>
# <li>بخش اول: بارگذاری داده </li>
# <li>بخش دوم: پیش پردازش </li>
# <li>بخش سوم: پیاده سازی شبکه و‌ آموزش شبکه</li>
# <li>بخش چهارم:مقایسه مدل‌ها</li>
# </ul>
# <li><b>سوال دوم - <span dir="rtl"> پیاده‌سازی طبقه‌بند اخبار با کمک شبکه عصبی و مدل Fasttext</span> </b></li>
# <ul>
# <li>بخش اول: بارگذاری داده و پیش پردازش </li>
# <li>بخش دوم: Embedding نمونه‌ها </li>
# <li>بخش سوم: پیاده سازی شبکه و‌ اموزش شبکه</li>
# <li>بخش چهارم: تحلیل نتایج</li>
# </ul>
# </div>
# </div>
# <div dir='rtl' style="line-height: 1.8; font-family: Vazir; font-size: 16px; margin-top: 20px; background-color: #e8eaf6; padding: 15px; border-radius: 8px; color:black">
# 💡 <b>نکات مهم:.</b>
# <br>
#  💡برای سوال اول باید تمامی بخش ها رو خودتان پیاده سازی کنید . حق استفاده از کتابخانه‌های اماده را ندارید.</div>
# </div>

# # <div style="text-align: center; direction: rtl; font-family: Vazir;"><h1 align="center" style="font-size: 24px; padding: 20px;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>سوال اول : پیاده سازی CBOW و Skip-Gram<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1></div>
# 

# <p dir="rtl" style="text-align: right; padding:30px; background-color:rgb(12, 12, 12); border-radius: 12px; color: white; font-family: Vazir;">
# در این سوال شما با پیاده سازی دو مدل CBOW و Skipgram آشنا میشوید.
# <p dir="rtl" style="padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# </p>
# </p>
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بارگذاری داده</div>
# 

# <div dir="rtl" style="text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# 
# ابتدا دیتاست زیر را دانلود کنید.
# <a>https://docs.pytorch.org/text/0.8.1/datasets.html#wikitext-2</a>
# </div>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# ۳ فایل txt که برای سه مجموعه داده Train, Valid, Test
# ذخیره شده باشد. 
# </p>

# In[1]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">پیش پردازش</div>
# 

# <div dir="rtl" style="text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# <p style="line-height: 1.8; text-align: right;">
# بعد از انکه داده را بارگذاری کردید. تمامی مراحل زیر را خودتان پیاده‌سازی کرده و پیش پردازش های گفته شده را انجام دهید
# <br>
# 
# بخش اول) یک تابعی که مراحل زیر را انجام دهد:
# ‌<br>
# ‍۱) تمامی کلمات را lowercase شود
# ‌<br>
# ۲)Special Characters حذف شوند
# ‌<br>
# ۳)کلمات در هر space (فاصله) از هم جدا و توکن شوند
# ‌<br>
# 
# 
# بخش دوم)  
# ۱)شمارش فرکانس (Frequency Counting): تعداد تکرار هر کلمه در تمام متن‌ها را حساب کنید
# ‌<br>
# ۲)فیلتر کردن کلمات نادر (Min Frequency Filtering): فقط کلماتی که بیشتر از min_freq بار تکرار شده‌اند را نگه دارید
# ‌<br>
# ۳)اضافه کردن توکن خاص <unk>: برای کلمات ناشناخته یا نادر
# ‌<br>
# ۴)ایجاد دیکشنری دوطرفه:
# ‌<br>
# word → index (string to index)
# ‌<br>
# index → word (index to string)
# 
# 
# بخش سوم)
# <br>
# ۱) تابعی بنویسید که از یک جمله، نمونه‌های آموزشی CBOW تولید کند
# <br>
# ۲)تابعی بنویسید که از یک جمله، نمونه‌های آموزشی Skip-gram تولید کند
# <br>
# <b>بخش سوم را توضیح بدهید</b>
# 
# <b>نکته: برای خوانایی در کد به جای پیاده سازی معمولی دو تابع،  میتوایند دو تابع گفته شده بالا را برای collate_fn پایتورچ پیاده‌سازی کنید.</b>
# 
# <a>https://discuss.pytorch.org/t/custom-collate-function/145823</a>
# <br>
# بخش چهارم)
# <br>
# در آخر داده پردازش شده را در dataloader لود کنید.
# </p>
# </div>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# به عنوان مثال
# <br>
# برای بخش اول 
# "Hello World! This is a TEST sentence, with 123 numbers."
# به عوان ورودی
# ["hello", "world", "this", "is", "a", "test", "sentence", "with", "123", "numbers"]
# شود
# 
# <br>
# برای بخش دوم
# <br>
# ورودی:
# texts = [
#     "the cat sat on the mat",
#     "the dog sat on the log",
#     "cats and dogs"
# ]
# min_freq = 2
# <br>
# خروجی:
# <br>
# Vocabulary:
# {
#     "<unk>": 0,
#     "the": 1,
#     "sat": 2,
#     "on": 3,
#     "cat": 4, 
#     "dog": 5
# }
# </p>
# 

# <div dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </div>
# 

# In[1]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">پیاده‌سازی شبکه و آموزش شبکه</div>
# 

# <div dir="rtl" style="text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# <p style="line-height: 1.8; text-align: right;">
# حال شبکه‌های SkipGram و CBOW را مانند مقاله (یا کتاب درسی) پیاده سازی کنید </p>
# شبکه را آموزش دهید و برای دادگان Train, Validation & Test نمودار خطا ترسیم کنید.
# </div>
# 

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">مقایسه شبکه‌ها</div>
# 

# <div dir="rtl" style="text-align: right; padding: 15px; background-color: #f5f5f5; border-radius: 12px; border: 2px solid #022216; font-family: Vazir; line-height: 1.8; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
# <h3 style="color: #022216; margin-top: 0;">دستورالعمل تحلیل شباهت واژگان</h3>
# 
# <ol style="padding-right: 20px;">
#     <li><strong>انتخاب واژه‌ها:</strong> ۵ واژه به دلخواه انتخاب نمایید.</li>
#     <li><strong>یافتن واژگان مشابه:</strong> برای هر واژه و هر مدل، با استفاده از معیار <em>Cosine Similarity</em>، ۵ واژه برتر مشابه را استخراج کنید.</li>
#     <li><strong>نمایش بصری:</strong> واژگان مشابه را با استفاده از روش <em>t-SNE</em> به صورت نمودار نمایش دهید (برای هر واژه و هر مدل یک نمودار جداگانه).</li>
#     <li><strong>تحلیل مقایسه‌ای:</strong> نمودارهای تولید شده را برای دو مدل مختلف با یکدیگر مقایسه کنید.</li>
# </ol>
# 
# <p style="color: #4b5563; font-size: 0.9em; margin-bottom: 0;">
#     نکته: در هر نمودار t-SNE می‌بایست واژه اصلی به همراه ۵ واژه مشابه آن نمایش داده شود.
# </p>
# </div>

# In[ ]:


# WRITE YOUR CODE HERE


# <div dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </div>
# 

#  # <div style="text-align: center; direction: rtl; font-family: Vazir;"><h1 align="center" style="font-size: 24px; padding: 20px;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>سوال دوم: پیاده‌سازی طبقه‌بند اخبار با کمک شبکه عصبی و مدل Fasttext<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1></div>

# <p dir="rtl" style="text-align: right; padding:30px; background-color:rgb(12, 12, 12); border-radius: 12px; color: white; font-family: Vazir;">
# در این سوال شما با کمک شبکه عصبی تمام متصل 
# (Fully Connected)
# یک طبقه‌بند متن پیاده‌سازی خواهید
# کرد.
# همچنین از مدل
# <a href="https://fasttext.cc/">Fasttext</a>
# برای Embed
# کردن متن‌ها استفاده می‌کنید.
# <p dir="rtl" style="padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# </p>
# </p>

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بارگذاری داده و پیش پردازش</div>
# 

# <div dir="rtl" style="text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
#     <ul>
#         <li>ابتدا دیتاست زیر دانلود کنید.
#     <br>
#     <a href="https://huggingface.co/datasets/SetFit/ag_news">link</a></li>
#         <li>
#             پس از دانلود مجموعه‌داده، 5000
#             نمونه از مجموعه‌داده آموزش و 2000 نمونه از مجموعه‌داده تست را به تصادف انتخاب کنید.
#         </li>
#         <li>
#             مجموعه 2000 نمونه‌ای را به دو مجموعه 1000 تایی تست و ارزیابی تقسیم کنید. همچنین از مجموعه 5000 تایی به عنوان مجموعه‌داده آموزش استفاده کنید.
#         </li>
#         <li>
#             در هنگام تشکیل مجموعه‌داده‌های جدید حتما توجه داشته‌باشید که توزیع داده‌ها در هر کلاس balanced باشد.
#         </li>
#         <li>
#         برای پیش‌پردازش متون تنها استفاده از lowercasing و 
#             حذف white space های اضافه 
#             کافی است.
#         </li>
#     </ul>
#     <b>سوال:</b> در این بخش به دلیل استفاده از مدل fasttext
#     برای embed کردن 
#         متون نیاز به پیش‌پردازش زیادی نداریم. چه ویژگی این مدل باعث می‌شود که ما از پیش پردازش بیشتر بی‌نیاز شویم؟
# </div>
# 

# <div dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </div>
# 

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# مجموعه‌داده‌های Train، Test و Validation
# </p>

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">Embedding نمونه‌ها</div>
# 

# <div dir="rtl" style="text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# در این بخش می‌خواهیم متون را برای طبقه‌بندی آماده کنیم.
#     <ol>
#         <li>
#             در ابتدا مدل pre-train شده 
#             <a href="https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip">wiki-news-300d-1M-subword</a>
#             را دانلود کنید و با کمک کتابخوانه fasttext 
#             آنرا load کنید.
#         </li>
#         <li>
#             در اسلاید ششم درس با مفهوم sentence embedding آشنا شده‌اید.
#             با کمک مدل fasttext 
#             تمامی متون موجود در مجموعه‌داده‌های train، test و validation
#             را embed کنید.
#         </li>
#     </ol>
# </div>
# 

# In[4]:


# WRITE YOUR CODE HERE


# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# بردارهای Embedding
# مجموعه‌داده‌های Train, Test, Validation
# </p>

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">پیاده سازی شبکه و‌ اموزش شبکه</div>
# 

# <div dir="rtl" style="text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# حال که متن‌ها را به فضای برداری نگاشت کرده‌ایم، مجموعه‌داده‌ها برای طبقه‌بندی با کمک شبکه عصبی آماده هستند.
# <ol>
#     <li>مجموعه‌داده ها را با کمک Dataloader pytorch
#     یا هر framework دیگری که استفاده می‌کنبد آماده کنید.
#     </li>
#     <li>
#     یک شبکه عصبی Fully Connected با معماری دلخواه برای آموزش طراحی کنید.
#     </li>
#     <li>
#     شبکه را حداقل به اندازه 30 ایپاک آموزش دهید.
#     </li>
# </ol>
# </div>
# 

# In[ ]:


# WRITE YOUR CODE HERE


# <div dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# <ul>
#     <li>
#     مدل آموزش داده شده
#     </li>
#     <li>
#     نمودار تغییرات دقت و loss
#         در هنگام آموزش برروی دو مجموعه‌داده
#         Train و Validation
#     </li>
# </ul>
# </div>

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">تحلیل نتایج</div>
# 

# <div dir="rtl" style="text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# پس از اتمام آموزش، موارد زیر را برای مجموعه داده test بدست آورید:
#     <ul>
#         <li>
#             ماتریس درهم‌ریختگی
#             (Confusion Matrix)
#         </li>
#         <li>
#             دقت
#         </li>
#         <li>
#             Macro-F1
#         </li>
#     </ul>
#     با توجه به متریک‌ها، به سوالات زیر جواب دهید:
#     <ul>
#         <li>مدل در تشخیص کدام کلاس‌ها بهترین و بدترین عملکرد را داشته‌است؟</li>
#         <li>مدل کدام کلاس‌ها را بیشترین دفعه با یکدیگر اشتباه گرفته‌است؟</li>
#         <li>سه نمونه از متونی که مدل به اشتباه آنها را برچسب زده‌است پیدا کنید و محتوای آنها را بررسی کنید.</li>
#     </ul>
# </div>
# 

# <div dir='rtl' style='background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111'>
# ✍️ <b>پاسخ تشریحی:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </div>
# 

# In[6]:


# WRITE YOUR CODE HERE


# <div dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# <ul>
#     <li>
#     متریک‌های گفته‌شده
#     </li>
#     <li>
#     تحلیل نتایج
#     </li>
# </ul>
# </div>

# # <h1 style="text-align: right;">**نکات مهم و قوانین تحویل**</h1>

# 
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
