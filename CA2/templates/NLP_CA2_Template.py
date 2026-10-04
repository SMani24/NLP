#!/usr/bin/env python
# coding: utf-8

# <div style="text-align: center; padding: 20px; font-family: Vazir;">
# <h1 align="center" style="font-size: 28px; color:rgb(64, 244, 202); width: 100%;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>تمرین ۲<br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1>
# <h2 style="color:rgb(90, 255, 184); font-size: 20px;">Logistic Regression and Naive Bayes</h2>
# <p align="center" style="color: #666; font-size: 16px;">فرهاد نصری - علیرضا زمانی</p>
# <p align="center" style="color: #666; font-size: 16px; margin-bottom: 30px;">farhadnasri999@gmail.com - shigzv@gmail.com</p>
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
# <div style="padding-right:100px">
# 📋 <b>ساختار تمرین:</b>
# <li><b>سوال اول - <span dir="ltr">Logistic Regression and Naive Bayes (from scratch)</span> (60)</b></li>
# <ul>
# <li>بخش اول: پیش پردازش داده‌های متنی و استفاده از Bag of Words</li>
# <li>بخش دوم: جداسازی داده‌های آموزش و آزمایش</li>
# <li>بخش سوم: پیاده‌سازی معیار‌های ارزیابی</li>
# <li>بخش چهارم: پیاده‌سازی Logistic Regression</li>
# <li>بخش پنجم: پیاده‌سازی Naive Bayes</li>
# <li>بخش ششم: تحلیل نتایج</li>
# </ul>
# <li><b>سوال دوم - <span dir="ltr">Logistic Regression and Naive Bayes (e.g. 
# sklearn)</span> (40)</b></li>
# <ul>
# <li>بخش اول: استخراج ویژگی‌های ساختاری</li>
# <li>بخش دوم: استخراج ویژگی‌های آماری و محتوایی</li>
# <li>بخش سوم: استخراج ویژگی‌های دلخواه</li>
# <li>بخش چهارم: ترکیب همه ویژگی‌های استخراج‌شده</li>
# </ul>
# </div>
# </div>
# <div dir='rtl' style="line-height: 1.8; font-family: Vazir; font-size: 16px; margin-top: 20px; background-color: #e8eaf6; padding: 15px; border-radius: 8px; color:black">
# 💡 <b>نکات مهم:</b>
# <br>
# در سوال دو مجاز به استفاده از کتابخانه‌های آماده برای مدل‌ها و متریک‌ها هستید ولی در سوال یک، همان‌طور که در متن سوال هم ذکر شده، باید از پایه پیاده‌سازی کنید.
# </div>
# </div>

# # <div style="text-align: center; direction: rtl; font-family: Vazir;"><h1 align="center" style="font-size: 24px; padding: 20px;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>سوال اول: پیاده‌سازی Logistic Regression و Naive Bayes <br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1></div>
# 

# <p dir="rtl" style="text-align: right; padding:30px; background-color:rgb(12, 12, 12); border-radius: 12px; color: white; font-family: Vazir;">
# در این سوال، شما باید این دو مدل پایه‌ای برای طبقه‌بندی متون را از صفر (بدون استفاده از کتابخانه‌های آماده) پیاده‌سازی کنید. هم‌چنین نیاز است، پیش‌پردازش داده‌ها و ارزیابی عملکرد مدل‌ها را نیز انجام دهید.
# 
# 
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش اول: پیش پردازش داده‌های متنی و استفاده از Bag of Words</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# در گام پیش‌پردازش، هدف ما آماده‌سازی داده‌های متنی برای استفاده در مدل‌های یادگیری ماشین است. از آن‌جا که مدل‌ها با داده‌های عددی کار می‌کنند، باید متن خام را به شکلی عددی تبدیل کنیم تا بتوانند الگوهای موجود در واژه‌ها را بیاموزند. در این بخش، قصد داریم با تبدیل متون به نمایش Bag of Words، هر ایمیل را بر اساس تعداد تکرار واژه‌هایش به یک بردار عددی تبدیل کنیم.
# <br>
# دیتاست emails_1.csv شامل دو ستون است:
# <br>
# text (متن ایمیل)
# <br>
# status (برچسب، مشخص‌کننده‌ی spam یا ham بودن ایمیل)
# <br>
# شما باید مراحل زیر را انجام دهید:
# <br>
# ۱. ستون status را به مقادیر عددی تبدیل کنید (spam → 1 و ham → 0).
# <br>
# ۲. متن‌ها را پاک‌سازی کنید: تمام کاراکترهای غیر الفبایی (اعداد، علائم و غیره) را حذف و همه‌ی حروف را کوچک (lowercase) کنید.
# <br>
# ۳. با استفاده از روش Bag of Words، فراوانی کلمات را محاسبه کنید و فقط ۱۵ کلمه‌ی پرتکرار را نگه دارید.
# <br>
# 💡 نکته: برای این بخش می‌توانید از CountVectorizer در کتابخانه‌ی sklearn استفاده کنید.
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# یک دیتافریم جدید که شامل ۱۵ ویژگی + ستون status است. (تمام ۱۰ ردیف این دیتافریم را نمایش دهید.)
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# از این مرحله به بعد، دیگر با فایل emails_1.csv و دیتافریم حاصل از آن کاری نخواهیم داشت. برای سهولت کار، فایل emails_2.csv در اختیار شما قرار گرفته است. این فایل، نتیجه‌ی همان فرایند پیش‌پردازش بر روی ۵۱۷۲ ایمیل واقعی است. به بیان ساده‌تر، در این مجموعه داده، هر ردیف نمایانگر یک ایمیل و هر ستون نشان‌دهنده‌ی فراوانی یکی از ۳۰۰۰ کلمه‌ی پرتکرار در کل داده‌ها است.
# در ادامه، مدل‌هایی که پیاده‌سازی می‌کنید باید روی این مجموعه داده، آموزش داده شوند. بنابراین، در این بخش فایل emails_2.csv را بخوانید و محتوای آن را نمایش دهید تا با ساختار داده آشنا شوید.
# </p>
# 
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# پنج ردیف اول دیتافریم مذکور را نمایش دهید.
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش دوم: جداسازی داده‌های آموزش و آزمایش</div>

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# داده‌ها را با نسبت ۸۰ درصد برای آموزش و ۲۰ درصد برای آزمایش تقسیم کنید. در انتها تعداد نمونه‌ها را چاپ کنید.
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# X_train, X_test, y_train, y_test<br>
# تعداد نمونه‌های X_train و X_test 
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش سوم: پیاده‌سازی معیار‌های ارزیابی</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# پیش از آن‌که وارد مرحله‌ی پیاده‌سازی و آموزش مدل‌های طبقه‌بندی شویم، لازم است معیارهایی برای سنجش عملکرد آن‌ها تعریف کنیم. این معیارها به ما کمک می‌کنند تا بفهمیم مدل تا چه اندازه در تشخیص درست نمونه‌های مثبت و منفی موفق عمل کرده است.
# <br>
# در این بخش، توابع زیر را برای حالت دودویی (binary classification) پیاده‌سازی کنید:
# <br>
# accuracy(y_true, y_pred) — نسبت پیش‌بینی‌های درست به کل نمونه‌ها
# <br>
# precision(y_true, y_pred) — درصد پیش‌بینی‌های مثبت که واقعاً مثبت بوده‌اند
# <br>
# recall(y_true, y_pred) — درصد نمونه‌های مثبت واقعی که مدل آن‌ها را درست شناسایی کرده است
# <br>
# f1_score(y_true, y_pred) — میانگین هارمونیک بین precision و recall، برای ایجاد توازن میان آن دو
# <br>
# هر تابع باید مقدار عددی متناظر با معیار مورد نظر را برگرداند.
# <br>
#  به توابع بالا دو ورودی زیر را بدهید و خروجی بگیرید:    y_true = [0, 1, 1, 0, 1] ---- y_pred = [0, 1, 0, 0, 1]
# <br>
# 💡 نکته: برای این بخش می‌توانید از  کتابخانه‌ی numpy استفاده کنید.
# </p>

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# مثال:
# <br>
# y_true = [0, 1, 1, 0, 1]
# <br>
# y_pred = [0, 1, 0, 0, 1]
# <br>
# print(accuracy(y_true, y_pred))  # 0.80
# <br>
# print(precision(y_true, y_pred)) # 1.00
# <br>
# print(recall(y_true, y_pred))    # 0.66
# <br>
# print(f1_score(y_true, y_pred))  # 0.80
# <br>
# </p>

# In[1]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش چهارم: پیاده‌سازی Logistic Regression</div>

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# کلاس Logistic Regression را به طور کامل پیاده سازی کنید.سپس مدل را بر روی داده‌های train آموزش دهید و در انتها دقت مدل را باتوجه به متریک های تعریف شده بر روی داده‌های test گزارش کنید.
# <br>
# 💡استفاده از هایپر پارامتر‌های مناسب مانند نرخ یادگیری و تعداد دوره‌های آموزش بر عهده خودتان است.
# <br>
# 💡میتوانید قبل از آموزش داده ها را نرمال سازی کنید.
# <br>
# 💡مجاز به استفاده از کتابخانه اماده نیستید.
# </p>

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# معیار‌های پیاده سازی شده را بر روی داده‌های test خروجی بگیرید و نتایج را چاپ کنید
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش پنجم: پیاده‌سازی Naive Bayes</div>

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# کلاس Multinomial Naive Bayes را به طور کامل پیاده سازی کنید. سپس مدل را بر روی داده‌های train آموزش دهید و در انتها دقت مدل را باتوجه به معیار های تعریف شده بر روی داده‌های test گزارش کنید.
# <br>
# - چرا از مدل‌های دیگر مانند Gaussian Naive Bayes استفاده نکردیم؟ آیا از مدل‌های دیگر Naive Bayes می‌توان استفاده کرد؟
# <br>
# 💡مجاز به استفاده از کتابخانه اماده نیستید.
# <br>
# 💡برای آشنایی با Naive Bayes می‌توانید به Appendix کتاب jurafsky مراجعه کنید.
# </p>

# <p style="direction: rtl; text-align: right; background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111">
# ✍️ <b>پاسخ تشریحی بخش پنجم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# معیار‌های پیاده سازی شده را بر روی داده‌های test خروجی بگیرید و نتایج را چاپ کنید
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;"> بخش ششم: تحلیل نتایج</div>

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# نتایج دو مدل پیاده سازی شده را با یکدیگر مقایسه کنید و تحلیلی بر نتایج داشته باشید.<br>
#     
# </p>

# In[ ]:





# <p style="direction: rtl; text-align: right; background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111">
# ✍️ <b>پاسخ تشریحی بخش ششم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# # <div style="text-align: center; direction: rtl; font-family: Vazir;"><h1 align="center" style="font-size: 24px; padding: 20px;">⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️<br>سوال دوم: استفاده از Logistic Regression و Naive Bayes <br>⚜️━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━⚜️</h1></div>
# 

# <p dir="rtl" style="text-align: right; padding:30px; background-color:rgb(12, 12, 12); border-radius: 12px; color: white; font-family: Vazir;">هدف این سوال، بررسی و مقایسه عملکرد مدل‌های Logistic Regression و Naive Bayes در تشخیص لینک‌های فیشینگ از لینک‌های قانونی (معتبر) است.
# شما باید با استفاده از مجموعه‌داده‌ی ارائه‌شده (آدرس‌های اینترنتی خام)، یک سری ویژگی‌های عددی از این URLها استخراج نمایید و تأثیر انتخاب ویژگی‌ها و نرمال‌سازی داده‌ها را بر عملکرد مدل‌ها بررسی کنید.<br>نکته: در این سوال مجاز به استفاده از کتاب‌خانه‌های آماده هستید.
# 

# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش اول: ویژگی‌های ساختاری</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# سه ویژگی زیر را از آدرس استخراج کنید:
# <br>
# - تعداد نقطه‌ها (nb_dots)
# <br>
# - تعداد اسلش‌ها (nb_slashes)
# <br>
# - تعداد خط ‌تیره‌ها (nb_hyphens)
# <br>
# سپس با این سه ویژگی مدل‌ها را روی هشتاد درصد داده‌ها (داده‌های آموزش) آموزش دهید و روی بیست درصد داده‌ها (داده‌های آزمون) مقایسه کنید
# <br>
# <br>
# - آیا لازم است این ویژگی‌ها را نرمال‌سازی کنید؟ چرا؟ اگر پاسخ شما «بله» است، نوع نرمال‌سازی را توضیح دهید و آن را اجرا کنید.
# <br>
# - به نظر شما از بین نسخه‌های مختلف Naive Bayes (GaussianNB یا MultinomialNB) کدام برای داده‌های شما مناسب‌تر است؟ دلیل خود را بنویسید و با آن مدل آزمایش را انجام دهید.
# </p>

# <p style="direction: rtl; text-align: right; background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111">
# ✍️ <b>پاسخ تشریحی بخش اول:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - جدولی (دیتافریم) شامل ۳ ویژگی استخراج‌شده + برچسب هدف (status)<br>
# - جدول مقایسه‌ای شامل Accuracy, Precision, Recall, F1-score برای هر مدل
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش دوم: ویژگی‌های آماری و محتوایی</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# سه ویژگی زیر را از آدرس استخراج کنید:
# <br>
# - طول کل آدرس (length_url)
# <br>
# - نسبت تعداد ارقام (0–9) به کل طول آدرس (ratio_digits_url)
# <br>
# - طولانی‌ترین کلمه (کاراکترهای الفبایی) در URL که با علائم جداساز (نقطه، علامت‌سوال، اسلش و...) از هم جدا شده‌اند. (longest_words_raw)
# <br>
# سپس با این سه ویژگی مدل‌ها را روی هشتاد درصد داده‌ها (داده‌های آموزش) آموزش دهید و روی داده‌های آزمون (بیست درصد داده‌ها) مقایسه کنید.
# <br>
# - به نظر شما استفاده از Naive Bayes برای این ویژگی‌ها منطقی است؟ توضیح دهید. کدام نسخه مناسب‌تر است؟ 
# </p>

# <p style="direction: rtl; text-align: right; background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111">
# ✍️ <b>پاسخ تشریحی بخش دوم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - جدولی (دیتافریم) شامل ۳ ویژگی استخراج‌شده + برچسب هدف (status)<br>
# - جدول مقایسه‌ای شامل Accuracy, Precision, Recall, F1-score برای هر مدل
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش سوم: ویژگی‌های خلاقانه</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# سه ویژگی به انتخاب خودتان از آدرس‌ها استخراج نمایید و مشابه دو بخش قبلی، مدل‌ها را آموزش دهید.
# برای هر ویژگی توضیح دهید چرا ممکن است در تشخیص فیشینگ مؤثر باشد؟
# </p>

# <p style="direction: rtl; text-align: right; background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111">
# ✍️ <b>پاسخ تشریحی بخش سوم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# <p dir='rtl' style="line-height: 2.0; text-align: right; font-family: Vazir; font-size: 16px; margin-top: 20px; color: white; background-color:rgb(0, 40, 30); padding: 30px; border-radius: 8px;">
# 🎯 <b>خروجی مورد انتظار:</b><br>
# - جدولی (دیتافریم) شامل ۳ ویژگی استخراج‌شده + برچسب هدف (status)<br>
# - جدول مقایسه‌ای شامل Accuracy, Precision, Recall, F1-score برای هر مدل
# </p>

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش چهارم: ترکیب همه ویژگی‌ها</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# تمام ۹ ویژگی (۳ ساختاری + ۳ آماری + ۳ خلاقانه) را با هم ترکیب کنید.
# سپس هر دو مدل را دوباره آموزش دهید و نتایج را ارزیابی کنید.
# <br>
# آیا ترکیب همه ویژگی‌ها باعث بهبود عملکرد مدل می‌شود؟
# <br>
# اگر خیر، به نظر شما دلیل آن چیست؟ (مثلاً تداخل بین ویژگی‌ها یا همبستگی بالا و...)
# </p>
# 

# <p style="direction: rtl; text-align: right; background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111">
# ✍️ <b>پاسخ تشریحی بخش چهارم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# In[ ]:


# WRITE YOUR CODE HERE


# ## <div style="text-align: center; direction: rtl; font-family: Vazir;">بخش پنجم: جمع‌بندی نهایی</div>
# 

# <p dir="rtl" style="line-height: 1.8; text-align: right; padding:10px; background-color:#6B7280;  border-radius: 12px; border: 2px solid rgb(2, 34, 22); font-family: Vazir;">
# نتیجه‌گیری نهایی خودتان از این سوال را ارائه دهید. برای مثال باید "حداقل" به سوال‌های زیر پاسخ دهید:
# <br>
# - کدام مدل برای این نوع داده بهتر عمل کرده است؟ چرا؟
# <br>
# - کدام نوع ویژگی بیشترین تأثیر را داشته است؟ کدام نوع کمترین؟ چرا؟
# <br>
# - نرمال‌سازی نیاز بوده است؟ اگر بله، برای هر دو مدل؟ چرا؟
# <br>
# - غیر از استخراح ویژگی از خود آدرس اینترنتی، چه رویکرد دیگری می‌توان اتخاذ نمود؟
# </p>
# 

# <p style="direction: rtl; text-align: right; background:#fffbe6; font-family: Vazir; border:1px dashed #f0ad4e; padding:12px; border-radius:8px; color:#111">
# ✍️ <b>پاسخ تشریحی بخش پنجم:</b><br>
# {{پاسخ_خود_را_اینجا_بنویسید}}
# </p>
# 

# # <h1 style="text-align: right;">**نکات مهم و قوانین تحویل**</h1>

# 
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

# In[ ]:




