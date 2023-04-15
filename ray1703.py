#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:36:15 2023

@author: jacqueline lee
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import math 


# change any parameter at the bottom 

plt.rcParams.update({'font.size': 14})

class Photon:
	def __init__(self, x, y, angle):
		self.angle= angle
		self.posi= np.array([x, y])
		self.line1= np.array([np.tan(self.angle*np.pi/180), -1, self.posi[1]-np.tan(self.angle*np.pi/180)*self.posi[0]]) # in the form of ax+by+c=0 
		self.pass1= 0
		self.intersect= []
		self.scatter_point=np.array([])
		self.scatter_angle= np.nan
		self.line2=np.array([]) # the line equation if the photon scatters 
		self.pass2= 0
		self.detect_point=np.array([])
		
		self.angle_list= np.array([])
		angle= np.linspace(-180, 180, 1000) 
		prob= self.Klein_Nishina(angle*np.pi/180)*100
		prob= np.asarray(prob, dtype = 'int')
		for i in range(len(prob)):
			angle_i= np.ones(int(prob[i]))*angle[i]
			self.angle_list= np.append(self.angle_list, angle_i)
		

	def plot_line1(self, end):
		x= np.linspace(0, end, 100)
		y= self.line1[0]*x +self.line1[2]
		return x, y
	
	def plot_line2(self, start):
		x= np.linspace(start, 30, 100)
		y= self.line2[0]*x +self.line2[2]
		return x, y
	
	def pass_through_or_not(self, x, y, r):
		dis= abs(self.line1[0]* x+ self.line1[1]*y+ self.line1[2])/np.sqrt(np.sum(self.line1**2))
		if dis> r: # does not pass through 
			return 0
		
		elif dis == r: # pass through but overlap with a point 
			m= self.line1[0]
			k= self.line1[2]
			x1= (-(2*m*k-2*x)+np.sqrt((2*m*k-2*x)**2-4*(1+m**2)*(x**2+k**2-r**2)))/(2*(1+m**2))
			self.intersect.append(np.array([x1, m*x1+k]))
			return 1
		
		else: # pass through and intersect two points 
			m= -self.line1[0]/self.line1[1]
			k= -self.line1[2]/self.line1[1]
			D= (2*m*k-2*x)**2-4*(1+m**2)*(x**2+k**2-r**2)
			if D > 0:
				x1= (-(2*m*k-2*x)-np.sqrt(D))/(2*(1+m**2))
				self.intersect.append(np.array([x1, m*x1+k]))
				x2= (-(2*m*k-2*x)+np.sqrt(D))/(2*(1+m**2))
				self.intersect.append(np.array([x2, m*x2+k]))
				return 1
			else:
				return 0
			
	
	def scatter_or_not(self, P= 1, dr= 0.5): 
		# P is the prob of the photon scatters 
		# P is approximately 0.01 for Al in the reality 
		if len(self.intersect)==2:
			x1= self.intersect[0]
			x2= self.intersect[1]
			r= math.dist(x1, x2)
			scatter_or_not= np.random.choice([0,1], p=[(1-P), P])
			
			if scatter_or_not == 1:
				unit_vector= (x2-x1) / np.linalg.norm(x2-x1)
				split= int(r/dr)
				if split >1:
					i= np.random.randint(1, split)
					self.scatter_point= x1+ unit_vector* i
				else:
					self.scatter_point= x1
				return True 
			
			else:
				self.pass1= 0
				return False

	
	def Klein_Nishina(self, theta):
		# gamma= hf/mc^2
		gamma= 1.29
		return (1+ np.cos(theta)**2)/(1+ gamma*(1-np.cos(theta))**2)*(1+(gamma**2 *(1-np.cos(theta))**2/((1+np.cos(theta)**2)*(1+gamma*(1-np.cos(theta))))))
	
	
	def prob_scatter_angle(self):
		# I assume it follows the Klein-Nishina formula 
		angle= np.linspace(-180, 180, 600) 
		prob= self.Klein_Nishina(angle*np.pi/180)*100
		prob= np.asarray(prob, dtype = 'int')
		scatter_angle= self.angle_list[np.random.randint(0, len(self.angle_list))]
		self.scatter_angle = scatter_angle 
		return scatter_angle
	
	
	def pass_through_Det_or_not(self, self_line, line, edge1, edge2):
		det= self_line[0] * line[1] - line[0] * self_line[1]
		if det == 0:
			# Equations are parallel or coincident
			return False
		x = (-self_line[2] * line[1] + line[2] * self_line[1]) / det
		y = (-self_line[0] * line[2] + line[0] * self_line[2]) / det

		if line[0]==0:
			if x > edge1[0] and x < edge2[0]:
				return (x,y)

		
		if  self.scatter_angle>0 and x > edge1[0] and x < edge2[0] and y > edge1[1] and y < edge2[1]:
			return (x,y)
		else:
			return False
		
	
class Rod:
	def __init__(self, x, y, radius):
		self.centre = np.array([x, y])
		self.r= radius 
		
class Det:
	def __init__(self, x, y, d, phi):
		phi= np.pi/2-phi*np.pi/180
		self.line= np.array([np.tan(phi), -1, y-x*np.tan(phi)])
		self.edge1= np.array([x-d/2*np.cos(phi), y-d/2*np.sin(phi)])
		self.edge2= np.array([x+d/2*np.cos(phi), y+d/2*np.sin(phi)])
		
		
	
	def plot_Det(self, start, end):
		x= np.linspace(start, end, 20)
		y= self.line[0]*x+ self.line[2]
		return x, y

class Simulation():
	def __init__(self, photons, rod, detector):
		self.photons= photons
		self.rod= rod
		self.det= detector
	
	def plot_dist(self):
		x= np.array([])
		y= np.array([])
		for photon in self.photons:
			if photon.pass2 ==1:
				x= np.append(x, photon.detect_point[0])
				y= np.append(y, photon.detect_point[1])
		plt.figure()
		plt.hist(x, bins=25)
		plt.title('x Position distribution')
	
	def find_angle(self):
		angle= np.array([])
		for photon in self.photons:
			if photon.pass1==1 and photon.pass2==1:
				angle= np.append(angle, photon.scatter_angle)
		plt.hist(angle, bins= 15)
		plt.xlabel('angle (degree)')
		plt.ylabel('counts')
		return angle
	

	
	def run(self, plot=False):
		count_Al=0

		for photon in self.photons:
			val= photon.pass_through_or_not(*self.rod.centre, self.rod.r)
			# photons that pass through the Al rod 
			if val==1:
				photon.pass1=1
				count_Al+=1
				scatter_or_not= photon.scatter_or_not()
				if scatter_or_not == True:
					angle= photon.prob_scatter_angle()
					photon.line2= np.array([np.tan(-angle*np.pi/180), -1, photon.scatter_point[1]-np.tan(-angle*np.pi/180)*photon.scatter_point[0]])
					detect= photon.pass_through_Det_or_not(photon.line2, self.det.line, self.det.edge1, self.det.edge2)
					if detect != False:
						photon.pass2=1
						photon.detect_point= np.array([detect[0], detect[1]])
			# photons that does not pass through the Al rod but are detected by the detector 

			else: 
				#print(self.det.line)
				detect= photon.pass_through_Det_or_not(photon.line1, self.det.line, self.det.edge1, self.det.edge2)
				if detect != False:
					photon.pass2=1
					photon.detect_point= np.array([detect[0], detect[1]])

		#self.plot_dist()
		#angle= self.find_angle()
		#print('no of photons pass through AL=', count_Al)
		if plot:
			fig, ax= plt.subplots()

			# plot out the paths of photons 
			for photon in self.photons:
				if photon.pass1 == 1:
					x, y= photon.plot_line1(photon.scatter_point[0])
					ax.plot(x, y, color= 'blue')
					
					if len(photon.line2) != 0:
						xs, ys= photon.plot_line2(photon.scatter_point[0])
						ax.plot(xs, ys, color='blue')
						
					if photon.pass2==1:
						xs, ys= photon.plot_line2(photon.scatter_point[0])
						ax.plot(x, y, color= 'blue')
						ax.plot(xs, ys, color='blue')
				else:
					x, y= photon.plot_line1(30)
					#ax.plot(x, y, color='blue')
					
			# plot out the Al rod
			circle = plt.Circle(self.rod.centre, self.rod.r, color='red', linewidth= 2, fill=False, label='Rod')
			ax.add_artist(circle)
			# plot out the detector 
			xd, yd= self.det.plot_Det(self.det.edge1[0], self.det.edge2[0])
			ax.plot(xd, yd, color='black', label='Detector')
			
			ax.grid()
			ax.set_aspect('equal')
			ax.set_xlim(-2, 30)
			ax.set_ylim(-10, 10)
			plt.legend()
			plt.show()
			
			

# change any parameter here 


Photons=[]
no_photons= 15# the number of photon 
d_source= 2 # the diameter of the hole on the Cs137 box 
rod= Rod(15, 0, 1.95/2) # Al rod: x_pos, y_pos, diameter 
# detector: x_pos, y_pos, diameter, angle from the x-axis 
angle_det= 30
detector= Det(15+ 15*np.cos(angle_det*np.pi/180), -15*np.sin(angle_det*np.pi/180), 1.98/2, angle_det) 

# generate the photons following the Gaussian from the Cs137 box 
# sigma of the Gaussian 
pos_y_i= np.random.uniform(-d_source/2, d_source/2, no_photons)

# maybe this part of simulation is wrong 
mu_angle= 0
sigma_angle= 0.3
angle= np.random.normal(mu_angle, sigma_angle, no_photons)


for i in range(no_photons):
	photon= Photon(0, pos_y_i[i], angle[i])
#	photon= Photon(0, 0, angle[i])
	Photons.append(photon)
	
	
	

i= Simulation(Photons, rod, detector)
i.run(plot= True) # say True if plot 





#%% Test 

def Klein_Nishina(theta):
	# gamma= hf/mc^2
	gamma= 1.29
	return (1+ np.cos(theta)**2)/(1+ gamma*(1-np.cos(theta))**2)*(1+(gamma**2 *(1-np.cos(theta))**2/((1+np.cos(theta)**2)*(1+gamma*(1-np.cos(theta))))))


theta= np.linspace(-180, 180, 30)
y= Klein_Nishina(theta*np.pi/180)


plt.plot(theta, y)




#%%

# calculate the prob of photon that does not scatter in Al

x= np.linspace(0, 1.95/2, 100)
mu= 0.137 # photon attteuation is 0.137 for Al
y= np.exp(-mu*x)
y= y/np.sum(y)
P_not_scatter= 1-y[-1]
print(P_not_scatter)

#%%

sa2=np.array([])
for photon in Photons:
	if photon.pass2==1:
		sa2= np.append(sa2, photon.scatter_angle)
		sa2= np.where(sa2>0, sa2, 0)
		sa2 = sa2[sa2 != 0]
		

		
plt.hist(sa2, bins=8)


#%%
import pandas as pd


df= pd.DataFrame({'90 degree': sa2})


df.to_csv('angular_dist_'+str(angle_det)+'.csv', index=False)




