#include <iostream>
#include <vector>

int main(){

    std::cout << 3 / 2 << std::endl;

    std::vector<int> temp;
    temp.push_back(1);
    temp.push_back(2);
    temp.push_back(3);
    temp.push_back(4);
    temp.push_back(5);

    
    std::cout << temp.size() << std::endl;
    //temp.erase(2);
    for(std::vector<int>::iterator it = temp.begin(); it != temp.end();it ++){
        if(*it == 4)
            it = temp.erase(it);
        if(*it == 3)
            it = temp.erase(it);
    }
    std::cout << temp.size() << std::endl;

    return 0;
}

