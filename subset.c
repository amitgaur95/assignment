#include<stdlib.h>

int check(int firstArr[], int secondArr[], int p, int q)
{
  int i, j;

  for(i=0;i<b;i++)
  {
    for(j=0; j<p; j++)
     {
        if(secondArr[j] == firstArr[i])
          break;
     }

     if(j == p)
       return 0;
  }
  return 1;
}

int main()
{
  int p, q, i, j;

  printf("Number of digits in array 1: ");
  scanf("%d",&p);  
  printf("Number of digits in array 2: ");
  scanf("%d",&q);

  int firstArr[100], secondArr[100];

  if(p<=10 && q<=5 && p>q)
  {
    printf("Input numbers in array 1: ");

    for(i=0; i<p; i++)
    {
      scanf("%d",firstArr[i]);
    }

    printf("Input numbers in array 2: ");

    for(i=0; i<p; i++)
    {
      scanf("%d",secondArr[i]);
    }

    if(check(firstArr, secondArr, p, q))
      printf("2nd array is not subset of 1st array");

    else
      printf("2nd array is the subset of 1st array");
  }  

  else{
    printf("Error \n");
  }
  return 0;
}