
# What's in a Network?
[<- Back to home page](index.md)

At the most basic of levels, a network is simply the linking of two or more computers, often to share information or resources. 

One example of this may consist of your phone, the computer you are using to read this site, and the modem that all these devices rely on to connect to the wider web. These devices are all part of your Local Area Network (LAN), which consists of all the devices that may be connected to your modem via a direct connection with a cable or wireless signal. The broader internet is an extension of this concept, consisting of innumerous networks that bridge and consolidate other sub-networks (just like your LAN) to form a large, interconnected tree that allows for all these devices to communicate with each other, from one branch to another. 

Each device communicates using packets, which are small snippets of data written in well-established protocols that are transmitted between sender and receiver. In the real world, these commonly have an upper limit size of about 1.5 kilobytes, and contain information about the connection between the sender and receiver. This includes metadata about the protocol, the IP Address of the senders and receivers, as well as other identifying information in relation to the packet and its relation to others.

# Common Network Issues

## Packet Loss
Packet loss is defined as any situation where a packet is dropped as it is being transmitted across the network. Often times, one can encounter packet loss due to reasons ranging from an unplugged ethernet cable to a bit being flipped due to the bombardment of background cosmic radiation on your router. Some likely causes for these include network congestion, hardware limitations, and software issues, to name a few. These result sometimes in routers discarding packets and not allowing them to continue. Fortunately, most modern connections have established protocol standards such as TCP that can inform machines in a connection whether a transmitted packet was sent properly or not. 

Packet loss rates in the range of 1-2.5% border on acceptable for most consumer use cases, while rates at around 4-6% are often noticeable in video calls. Any rates above that result in generally poor connections that border on being unusable. 

## Latency
Latency, oftentimes called lag, is the term used to describe any delay in communication over a network connection. Latency is measured in milliseconds, and one of the main causes is the distance between nodes. The sheer amount of distance that data often has to travel on the internet between sender and receiver can add significant amounts of unwanted delay, possibly up to hundreds of milliseconds. Issues arise in high latency connections, in which packets take a long time to reach their intended destinations. This commonly results in lack of responsiveness.  While much engineering has gone into making connections as fast and reliable as possible, latency is still very common due to the physical limitations of our current gen hardware.  

Latencies of about 150 ms begin to border on moderately acceptable for Voice Over IP (VoIP) audio calls, while anything over 300 ms is often classified as unusable for calling. For real-time gaming and most other applications, 20-40 ms is typically optimal, while latencies of up to 100 ms are still considered very usable.


# Sources and Further Reading:
- “Chapter 1: What Is a Network?” Florida Center for Instructional Technology, Florida Dept of Education, https://fcit.usf.edu/network/chap1/chap1.htm.
- "What Is Network Packet Loss?" IR Media, https://www.ir.com/guides/what-is-network-packet-loss
- "Network Latency - Common Causes and Best Solutions" IR Media, https://www.ir.com/guides/what-is-network-latency
- "What is Latency?" Optimum, https://www.optimum.com/internet/what-latency
- "What is TCP/IP?" Cloudflare, https://www.cloudflare.com/learning/ddos/glossary/tcp-ip/

[<- Back to home page](index.md)