# Popularity Prediction for News Article Entries on Facebook #

### What is this repository for? ###

* Quick summary

	* search tweets back into the past and store them in JSON format
 
* Version

	* v1.0 : basic function for search and store in JSON

### How do I use it? ###

* Configuration
	* open twitter-gathering.py in your favorite editor
	* change line 21 with your own directory and file name
	* change line 26 with your own search query
		* for query format, https://dev.twitter.com/rest/public/search 
	* change line 37~40 with your own authentication information
		* to obtain your authentication info, https://apps.twitter.com/app/new
* Dependencies
	* tweepy
		* pip install tweepy
		* you might need to use virtualenv in case you do not have sudo access
* How to run
	* for 1-time search
		* python twitter-gathering.py
	* for continuous searching(automatic running)
		* use crontab in Linux

### Example ###

* example output data

``` javascript
{
    "text": "L'empresa de seguretat de l'assass\u00ed hom\u00f2fob d'Orlando, G4S, va ser denunciada per col\u00b7laborar amb la tortura https://t.co/J0VQnKGWXl", 
    "created_at": "2016-06-13 15:56:23", 
    "source": "Twitter Web Client", 
    "in_reply_to_user_id": null, 
    "geo": null, 
    "id": 742385132217217024, 
    "user": {
        "utc_offset": 7200, 
        "favourites_count": 983, 
        "description": "Peri\u00f2dic Popular dels Pa\u00efsos Catalans", 
        "friends_count": 1879, 
        "created_at": "2009-11-27 12:58:04", 
        "time_zone": "Paris", 
        "followers_count": 17123, 
        "location": "Pa\u00efsos Catalans", 
        "geo_enabled": true, 
        "id": 92963933
    }
},
{
    "text": "RT @julienbellver: \u00ab\u00a0Orlando, une haine si particuli\u00e8re, le racisme anti-homosexuel\u00a0\u00bb @lemondefr https://t.co/44u5cIthcA", 
    "created_at": "2016-06-13 15:56:23", 
    "source": "Twitter for Android", 
    "in_reply_to_user_id": null, 
    "geo": null, 
    "id": 742385132158496768, 
    "user": {
        "utc_offset": null, 
        "favourites_count": 180, 
        "description": "Communicant/Medias/Sport/Cayenne", 
        "friends_count": 1401, 
        "created_at": "2014-07-15 14:51:40", 
        "time_zone": null, 
        "followers_count": 359, 
        "location": "Guyane", 
        "geo_enabled": false, 
        "id": 2712758357
    }
},
```

### Note ###

* output is NOT in the proper json format
	* output file ends with `,`
	* this is for appending newer data at the end of output file
		* this should be addressed in later version


### Who do I talk to? ###


