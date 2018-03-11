<p align="center">
  <img src="./sys.png" alt="Recommendation Engine"
       width="904" height="350">
</p>

<h1 align="center"> Types of Recommendation Engine </h1> <br>

Before taking a look at different types of recommendation engine, let’s take a step back and see if we can make some intuitive recommendations. Consider the following cases:

## Case 1: Recommend the most popular items
<p align="center">
  <img src="https://media.giphy.com/media/3bbcwarXienQzzWo5I/source.mp4" alt="Case-1"
       >
</p>
A simple approach could be to recommend items which are liked by the most number of users. This is a blazing fast and dirty approach and thus has a major drawback. The thing is, there is no personalization involved with this approach.

Basically, the most popular items would be identical for each user since popularity is against the entire user pool. So, everybody will see the same result. It sounds like, ‘a website recommending you to buy a microwave just because it’s been liked by other users and doesn’t care if you are even interested in buying or not’.

Surprisingly, such approach still works in services like news portals. When you login to say bbc news, you’ll see a column of “Popular News” which is subdivided into sections and the most read articles of each sections are displayed. This approach can work in this case because sections are divided allowing for a user to look at a particular section of interest.
At a time, there are only a few hot topics and there is a high chance that a user wants to read news that is read by the majority of other users.

## Case 2: Using a classifier to make recommendation
We already know lots of classification algorithms. Let’s see how we can use the same technique to make recommendations. Classifiers are parametric solutions so we just need to define some parameters (features) of the user and the item. The outcome can be  `1` if the user likes it or  `0` otherwise. This might work out in some cases because of following advantages:

1. Incorporates personalization
It can work even if the user’s past history is short or not available. However, this has some major drawbacks as well because it is not commonly practiced.

1. The features might not be available or even if they are, they may not be sufficient to make a good classifier. The results of this is that as the number of users and items grow, making a good classifier will become exponentially difficult.


## Case 3: Recommendation Algorithms
Now let’s come to the special class of algorithms which are tailor-made for solving the recommendation problem. There are typically two types of algorithms – Content Based and Collaborative Filtering. You should refer to our previous article to get a complete sense of how they work. I’ll give a short recap here.

## Content based algorithms
Idea: If you like an item then you will also like a “similar” item. This is based on the similarity of the items being recommended.
It generally works well when it’s easy to determine the context/properties of each item. For instance, when we are recommending the same kind of item like a movie or song.

## Collaborative filtering algorithms
Idea: If person `A` likes item `1, 2, 3` and `B` likes `2,3,4` then they have similar interests. Therefore, person `A` should like item `4`, and person `B` should like item `1`.
This algorithm is entirely based on the past behaviour of other persons and not on the context. This makes it one of the most commonly used algorithm as it is not dependent on any additional information. 
For instance, product recommendations by e-commerce player like Amazon and merchant recommendations by banks like American Express.

Further, there are several types of collaborative filtering algorithms: User-User, Item-Item, and et al.

## User-User Collaborative filtering
Here we find look-a-like customers (based on similarity) and offer products to the first customer that are similar to a look-a-like customer’s past behaviour. This algorithm is very effective but takes a lot of time and resources. It requires us to compute every customer information pair which takes time. Therefore, for big base platforms, this algorithm is hard to implement without a very strong parallelizable system.

## Item-Item Collaborative filtering
It is quite similar to the previous algorithm, but instead of finding alike customers, we try finding an alike item. Once we have the item look-a-like matrix, we can easily recommend alike items to customer who have purchased any item from the store. This algorithm is far less resource consuming than user-user collaborative filtering. Hence, for a new customer the algorithm takes far less time than the User-User Collaborate filtering as we don’t need all similarity scores between customers. And with fixed number of products the Product-Product look-a-like matrix is fixed over time.

Other simpler algorithms: There are other approaches like market basket analysis, which generally do not have high predictive power than the algorithms described above.

## Hybrid recommendation systems 
Hybrid recommendation systems combine both collaborative and content-based approaches. They help improve recommendations that are derived from sparse datasets. The Netflix algorithm is a prime example of a hybrid recommender.
